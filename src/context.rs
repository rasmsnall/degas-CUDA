use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;

use crate::{DeviceInfo, Error, GpuBuffer, GpuConfig, GpuKernel, GpuModule, KernelLaunch, LaunchConfig, Result};

/// Everything goes through here.
///
/// `GpuContext` opens a connection to a GPU and holds the default stream that
/// memory transfers and kernel launches run on. For most programs one context
/// on device 0 is all you need.
///
/// It's `Send + Sync`, so you can wrap it in an `Arc` and share it across
/// threads. If you're doing that and want overlapping GPU work, give each
/// thread its own stream via [`new_stream`][GpuContext::new_stream].
///
/// ```no_run
/// use degas_cuda::GpuContext;
///
/// let ctx = GpuContext::new(0)?;
/// println!("{}", ctx.info()?);
/// # Ok::<(), degas_cuda::Error>(())
/// ```
///
/// To customise which GPU or change launch settings, load a config:
///
/// ```no_run
/// use degas_cuda::{GpuConfig, GpuContext};
///
/// let config = GpuConfig::load_or_default("gpu_config.json")?;
/// let ctx = GpuContext::with_config(config)?;
/// # Ok::<(), degas_cuda::Error>(())
/// ```
pub struct GpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: GpuConfig,
}

impl GpuContext {
    /// Open device `device_id` with default settings.
    ///
    /// `0` picks the first GPU. Use [`device_count`][GpuContext::device_count]
    /// if you need to know how many are available.
    pub fn new(device_id: usize) -> Result<Self> {
        Self::with_config(GpuConfig { device_id, ..Default::default() })
    }

    /// Open a GPU using the settings in `config`.
    ///
    /// The device that gets opened is `config.device_id`.
    pub fn with_config(config: GpuConfig) -> Result<Self> {
        let ctx = CudaContext::new(config.device_id)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, config })
    }

    /// How many CUDA GPUs are visible to this process.
    pub fn device_count() -> Result<usize> {
        CudaContext::device_count()
            .map(|n| n as usize)
            .map_err(Error::Driver)
    }

    /// Name, compute capability, and memory size for this device.
    pub fn info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            ordinal: self.ctx.ordinal(),
            name: self.ctx.name()?,
            compute_capability: self.ctx.compute_capability()?,
            total_memory_bytes: self.ctx.total_mem()?,
        })
    }

    /// The config this context was opened with.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    // ── Memory ───────────────────────────────────────────────────────────────

    /// Copy a host slice to GPU memory and return a buffer wrapping it.
    ///
    /// ```no_run
    /// # let ctx = degas_cuda::GpuContext::new(0)?;
    /// let buf = ctx.upload(&[1.0_f32, 2.0, 3.0])?;
    /// # Ok::<(), degas_cuda::Error>(())
    /// ```
    pub fn upload<T: DeviceRepr>(&self, data: &[T]) -> Result<GpuBuffer<T>> {
        let inner = self.stream.clone_htod(data)?;
        Ok(GpuBuffer { inner })
    }

    /// Allocate `len` elements of GPU memory, all zeroed out.
    /// Works for any primitive numeric type.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<GpuBuffer<T>> {
        let inner = self.stream.alloc_zeros::<T>(len)?;
        Ok(GpuBuffer { inner })
    }

    /// Copy a GPU buffer back to the CPU and return it as a `Vec`.
    /// Blocks until the copy finishes.
    pub fn download<T: DeviceRepr>(&self, buf: &GpuBuffer<T>) -> Result<Vec<T>> {
        self.stream.clone_dtoh(&buf.inner).map_err(Error::Driver)
    }

    /// Overwrite an existing GPU buffer with new data from the CPU.
    /// The source and destination must have the same number of elements —
    /// this doesn't resize anything.
    ///
    /// Returns [`Error::SizeMismatch`] if the lengths differ.
    pub fn upload_into<T: DeviceRepr>(&self, src: &[T], dst: &mut GpuBuffer<T>) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::SizeMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        self.stream
            .memcpy_htod(src, &mut dst.inner)
            .map_err(Error::Driver)
    }

    // ── PTX loading ──────────────────────────────────────────────────────────

    /// Load a PTX module from a [`Ptx`] value directly.
    ///
    /// Use this if you have a `Ptx` from `cudarc::nvrtc::compile_ptx`, or if
    /// you need `Ptx::from_binary` for a CUBIN. For the common case of loading
    /// a `.ptx` text file, [`load_ptx_src`][Self::load_ptx_src] or
    /// [`load_ptx_file`][Self::load_ptx_file] is simpler.
    pub fn load_module(&self, ptx: Ptx) -> Result<GpuModule> {
        let inner = self.ctx.load_module(ptx)?;
        Ok(GpuModule { inner })
    }

    /// Load a PTX module from a string.
    ///
    /// The easiest way to ship a PTX kernel with your binary is
    /// `include_str!("../kernels/vadd.ptx")` — the file gets baked in at
    /// compile time.
    ///
    /// ```no_run
    /// # let ctx = degas_cuda::GpuContext::new(0)?;
    /// let module = ctx.load_ptx_src(include_str!("../kernels/vadd.ptx"))?;
    /// # Ok::<(), degas_cuda::Error>(())
    /// ```
    pub fn load_ptx_src(&self, src: &str) -> Result<GpuModule> {
        self.load_module(Ptx::from_src(src))
    }

    /// Load a PTX module from a file path. The file is read at runtime.
    pub fn load_ptx_file(&self, path: impl AsRef<Path>) -> Result<GpuModule> {
        self.load_module(Ptx::from_file(path))
    }

    // ── Kernel launch ────────────────────────────────────────────────────────

    /// Start building a kernel launch on the default stream.
    ///
    /// If `sync_on_launch` is set in your config, every `execute()` on the
    /// returned builder will block until the kernel finishes.
    ///
    /// ```no_run
    /// use degas_cuda::{GpuContext, LaunchConfig};
    ///
    /// fn run(ctx: &GpuContext, module: &degas_cuda::GpuModule) -> degas_cuda::Result<()> {
    ///     let n = 1024_i32;
    ///     let a = ctx.upload(&vec![1.0_f32; n as usize])?;
    ///     let b = ctx.upload(&vec![2.0_f32; n as usize])?;
    ///     let mut c = ctx.alloc_zeros::<f32>(n as usize)?;
    ///
    ///     let kernel = module.get_kernel("vector_add")?;
    ///     let mut launch = ctx.prepare(&kernel);
    ///     launch.arg_buf(&a).arg_buf(&b).arg_buf_mut(&mut c).arg_val(&n);
    ///     unsafe { launch.execute(ctx.launch_config_1d(n as u32))? };
    ///     Ok(())
    /// }
    /// ```
    pub fn prepare<'a>(&'a self, kernel: &'a GpuKernel) -> KernelLaunch<'a> {
        kernel.on(&self.stream, self.config.sync_on_launch)
    }

    /// Build a 1-D [`LaunchConfig`] sized for `num_elements` threads, using
    /// the `block_size` from this context's config.
    ///
    /// The grid is `ceil(num_elements / block_size)` blocks. With the default
    /// `block_size` of 256, launching over 1024 elements gives a 4×256 grid.
    ///
    /// ```no_run
    /// # let ctx = degas_cuda::GpuContext::new(0)?;
    /// let cfg = ctx.launch_config_1d(1024);
    /// # Ok::<(), degas_cuda::Error>(())
    /// ```
    pub fn launch_config_1d(&self, num_elements: u32) -> LaunchConfig {
        let block = self.config.block_size;
        let grid = num_elements.div_ceil(block);
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    // ── Synchronisation ──────────────────────────────────────────────────────

    /// Block until all work queued on the default stream has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(Error::Driver)
    }

    // ── Advanced access ──────────────────────────────────────────────────────

    /// Create a second stream on the same device. Useful if you want to overlap
    /// transfers and kernel execution, or run work from multiple threads
    /// concurrently.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        self.ctx.new_stream().map_err(Error::Driver)
    }

    /// The default stream this context uses for everything.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The underlying cudarc `CudaContext`, in case you need something this
    /// wrapper doesn't expose.
    pub fn raw(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}
