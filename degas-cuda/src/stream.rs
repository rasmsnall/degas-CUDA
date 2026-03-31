use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::{Error, GpuBuffer, GpuKernel, KernelLaunch, LaunchConfig, PinnedBuffer, Result};

/// An explicit CUDA stream for pipelining GPU work.
///
/// The default stream on a `GpuContext` serializes everything. That's fine
/// for simple programs, but if you want to overlap a host-to-device transfer
/// with a kernel running on the same device, you need two independent streams.
///
/// Get one with [`GpuContext::stream_handle`][crate::GpuContext::stream_handle].
/// Multiple streams can run concurrently as long as they don't write the same
/// buffer at the same time.
///
/// ```no_run
/// # let ctx = degas_cuda::GpuContext::new(0)?;
/// // Start a transfer on stream_a while stream_b runs a kernel.
/// let stream_a = ctx.stream_handle()?;
/// let stream_b = ctx.stream_handle()?;
///
/// let data: Vec<f32> = vec![1.0; 65_536];
/// let buf = stream_a.upload(&data)?;  // queued, returns immediately
///
/// // ... launch a kernel on stream_b here ...
///
/// stream_a.synchronize()?;            // wait for the upload to land
/// # Ok::<(), degas_cuda::Error>(())
/// ```
///
/// For maximum transfer throughput, combine streams with pinned host memory:
///
/// ```no_run
/// # let ctx = degas_cuda::GpuContext::new(0)?;
/// let stream = ctx.stream_handle()?;
/// let host = ctx.pinned_from(&vec![0.0_f32; 65_536])?;
///
/// // Direct DMA from pinned memory, no staging copy.
/// let buf = stream.upload_pinned(&host)?;
/// stream.synchronize()?;
/// # Ok::<(), degas_cuda::Error>(())
/// ```
pub struct GpuStream {
    pub(crate) inner: Arc<CudaStream>,
    // Keeps the context alive for as long as this stream exists.
    _ctx: Arc<CudaContext>,
}

impl GpuStream {
    pub(crate) fn new(stream: Arc<CudaStream>, ctx: Arc<CudaContext>) -> Self {
        Self { inner: stream, _ctx: ctx }
    }

    // uploads

    /// Copy a host slice to GPU memory, queued on this stream.
    /// Returns before the transfer completes. Call [`synchronize`][Self::synchronize]
    /// when you need the data to be on the GPU.
    pub fn upload<T: DeviceRepr>(&self, data: &[T]) -> Result<GpuBuffer<T>> {
        let inner = self.inner.clone_htod(data)?;
        Ok(GpuBuffer { inner })
    }

    /// Copy a pinned host buffer to GPU memory, queued on this stream.
    ///
    /// Faster than `upload` because the GPU DMA's directly from page-locked
    /// memory. No internal staging copy is needed. Use this with
    /// [`GpuContext::pinned_from`][crate::GpuContext::pinned_from] or
    /// [`GpuContext::alloc_pinned`][crate::GpuContext::alloc_pinned].
    pub fn upload_pinned<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        pinned: &PinnedBuffer<T>,
    ) -> Result<GpuBuffer<T>> {
        let inner = self.inner.clone_htod(&pinned.inner)?;
        Ok(GpuBuffer { inner })
    }

    // downloads

    /// Copy a GPU buffer back to the CPU as a `Vec`. Blocks until done.
    pub fn download<T: DeviceRepr>(&self, buf: &GpuBuffer<T>) -> Result<Vec<T>> {
        self.inner.clone_dtoh(&buf.inner).map_err(Error::Driver)
    }

    /// Copy a GPU buffer into a pinned host buffer, queued on this stream.
    ///
    /// The transfer is asynchronous. The pinned buffer's data is not valid
    /// until you call [`synchronize`][Self::synchronize]. After that, read
    /// the result with [`PinnedBuffer::as_slice`][crate::PinnedBuffer::as_slice].
    ///
    /// Returns [`Error::SizeMismatch`] if `dst` is smaller than `src`.
    pub fn download_into_pinned<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        src: &GpuBuffer<T>,
        dst: &mut PinnedBuffer<T>,
    ) -> Result<()> {
        if dst.len() < src.len() {
            return Err(Error::SizeMismatch { src: src.len(), dst: dst.len() });
        }
        self.inner
            .memcpy_dtoh(&src.inner, &mut dst.inner)
            .map_err(Error::Driver)
    }

    // allocation

    /// Allocate zeroed GPU memory on this stream's device.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<GpuBuffer<T>> {
        let inner = self.inner.alloc_zeros::<T>(len)?;
        Ok(GpuBuffer { inner })
    }

    // kernel launch

    /// Start building a kernel launch on this stream.
    ///
    /// Same as [`GpuContext::prepare`][crate::GpuContext::prepare] but the
    /// kernel runs on this stream instead of the default one.
    pub fn prepare<'a>(&'a self, kernel: &'a GpuKernel) -> KernelLaunch<'a> {
        kernel.on(&self.inner, false)
    }

    // sync

    /// Block until all work queued on this stream has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(Error::Driver)
    }

    // launch configs — mirrors GpuContext for convenience

    /// 1-D launch config for `num_elements` threads with a 256-thread block.
    pub fn launch_config_1d(&self, num_elements: u32) -> LaunchConfig {
        let block: u32 = 256;
        LaunchConfig {
            grid_dim: (num_elements.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 2-D launch config for a `width x height` grid with a 16x16 block.
    pub fn launch_config_2d(&self, width: u32, height: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (width.div_ceil(16), height.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 3-D launch config for a `width x height x depth` volume with an 8x8x4 block.
    pub fn launch_config_3d(&self, width: u32, height: u32, depth: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (width.div_ceil(8), height.div_ceil(8), depth.div_ceil(4)),
            block_dim: (8, 8, 4),
            shared_mem_bytes: 0,
        }
    }

    /// The raw cudarc stream, for cases where you need something not
    /// exposed here directly.
    pub fn raw(&self) -> &Arc<CudaStream> {
        &self.inner
    }
}
