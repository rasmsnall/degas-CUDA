use degas_cuda::{DeviceInfo, DeviceRepr, GpuConfig, GpuContext, ValidAsZeroBits};

use crate::{Canvas, GpuVec, Result, Volume};

/// The entry point for monet-cuda.
///
/// Open a `Session` instead of a `GpuContext` when you want the higher-level
/// shaped buffer types ([`Canvas`], [`Volume`], [`GpuVec`]) and shorter setup
/// code. For anything `Session` doesn't cover, like loading PTX, multi-stream
/// work, or raw kernel launches, call [`context`][Session::context] and use
/// `degas_cuda` directly.
///
/// ```no_run
/// use monet_cuda::Session;
///
/// let gpu = Session::new(0)?;
/// println!("{}", gpu.info()?);
/// # Ok::<(), monet_cuda::Error>(())
/// ```
pub struct Session {
    ctx: GpuContext,
}

impl Session {
    /// Open device `device_id` with default settings. `0` picks the first GPU.
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { ctx: GpuContext::new(device_id)? })
    }

    /// Open a GPU using a [`GpuConfig`], useful when you want to load
    /// settings from a JSON file or pick a specific device.
    ///
    /// ```no_run
    /// use degas_cuda::GpuConfig;
    /// use monet_cuda::Session;
    ///
    /// let config = GpuConfig::load_or_default("gpu.json")?;
    /// let gpu = Session::with_config(config)?;
    /// # Ok::<(), monet_cuda::Error>(())
    /// ```
    pub fn with_config(config: GpuConfig) -> Result<Self> {
        Ok(Self { ctx: GpuContext::with_config(config)? })
    }

    /// Name, compute capability, and memory for this device.
    pub fn info(&self) -> Result<DeviceInfo> {
        self.ctx.info()
    }

    /// Block until all queued GPU work finishes.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize()
    }

    /// The underlying [`GpuContext`] for PTX loading, kernel launches, and
    /// anything else this session doesn't expose directly.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    // 1-D

    /// Upload a host slice to GPU memory as a 1-D [`GpuVec`].
    pub fn vec_from<T: DeviceRepr>(&self, data: &[T]) -> Result<GpuVec<T>> {
        Ok(GpuVec { buf: self.ctx.upload(data)? })
    }

    /// Allocate a zeroed 1-D [`GpuVec`] of `len` elements.
    pub fn vec_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<GpuVec<T>> {
        Ok(GpuVec { buf: self.ctx.alloc_zeros(len)? })
    }

    /// Copy a 1-D [`GpuVec`] back to the CPU.
    pub fn download_vec<T: DeviceRepr>(&self, vec: &GpuVec<T>) -> Result<Vec<T>> {
        self.ctx.download(&vec.buf)
    }

    // 2-D

    /// Upload a host slice as a 2-D [`Canvas`] with the given dimensions.
    ///
    /// `data.len()` must equal `width * height`.
    pub fn canvas_from<T: DeviceRepr>(
        &self,
        data: &[T],
        width: usize,
        height: usize,
    ) -> Result<Canvas<T>> {
        Ok(Canvas { buf: self.ctx.upload_2d(data, width, height)? })
    }

    /// Allocate a zeroed 2-D [`Canvas`] of `width × height` elements.
    pub fn canvas_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        width: usize,
        height: usize,
    ) -> Result<Canvas<T>> {
        Ok(Canvas { buf: self.ctx.alloc_zeros_2d(width, height)? })
    }

    /// Copy a [`Canvas`] back to the CPU as a flat row-major `Vec`.
    pub fn download_canvas<T: DeviceRepr>(&self, canvas: &Canvas<T>) -> Result<Vec<T>> {
        self.ctx.download_2d(&canvas.buf)
    }

    // 3-D

    /// Upload a host slice as a 3-D [`Volume`] with the given dimensions.
    ///
    /// `data.len()` must equal `width * height * depth`.
    pub fn volume_from<T: DeviceRepr>(
        &self,
        data: &[T],
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<Volume<T>> {
        Ok(Volume { buf: self.ctx.upload_3d(data, width, height, depth)? })
    }

    /// Allocate a zeroed 3-D [`Volume`] of `width × height × depth` elements.
    pub fn volume_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<Volume<T>> {
        Ok(Volume { buf: self.ctx.alloc_zeros_3d(width, height, depth)? })
    }

    /// Copy a [`Volume`] back to the CPU as a flat `Vec`.
    pub fn download_volume<T: DeviceRepr>(&self, vol: &Volume<T>) -> Result<Vec<T>> {
        self.ctx.download_3d(&vol.buf)
    }
}
