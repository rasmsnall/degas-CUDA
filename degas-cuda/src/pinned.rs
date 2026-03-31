use cudarc::driver::{CudaContext, DeviceRepr, PinnedHostSlice, ValidAsZeroBits};

use crate::Result;

/// Page-locked host memory for fast GPU transfers.
///
/// Ordinary heap memory (a `Vec`) can't be DMA'd directly by the GPU. The
/// CUDA driver has to copy it into an internal pinned staging buffer first,
/// then do the actual transfer. That means every upload pays for two copies.
///
/// `PinnedBuffer` allocates the host memory in page-locked form from the
/// start. The GPU can DMA straight into or out of it, cutting transfer time
/// roughly in half and allowing transfers to overlap with kernel execution on
/// a second stream.
///
/// Create one with [`GpuContext::alloc_pinned`][crate::GpuContext::alloc_pinned]
/// or [`GpuContext::pinned_from`][crate::GpuContext::pinned_from], then pass it
/// to [`GpuContext::upload_pinned`][crate::GpuContext::upload_pinned] or
/// [`GpuStream::upload_pinned`][crate::GpuStream::upload_pinned].
///
/// ```no_run
/// # let ctx = degas_cuda::GpuContext::new(0)?;
/// // Fill a pinned buffer and upload it in one call.
/// let buf = ctx.pinned_from(&[1.0_f32, 2.0, 3.0])?;
/// let gpu_buf = ctx.upload_pinned(&buf)?;
///
/// // Or allocate first and fill manually.
/// let mut host = ctx.alloc_pinned::<f32>(1024)?;
/// host.as_slice_mut()
///     .iter_mut()
///     .enumerate()
///     .for_each(|(i, x)| *x = i as f32);
/// let gpu_buf = ctx.upload_pinned(&host)?;
/// # Ok::<(), degas_cuda::Error>(())
/// ```
pub struct PinnedBuffer<T: DeviceRepr + ValidAsZeroBits> {
    pub(crate) inner: PinnedHostSlice<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits> PinnedBuffer<T> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Total size in bytes.
    pub fn num_bytes(&self) -> usize {
        self.inner.num_bytes()
    }

    /// Read the buffer contents. Waits for any in-flight GPU writes to finish
    /// before returning the slice.
    pub fn as_slice(&self) -> crate::Result<&[T]> {
        self.inner.as_slice().map_err(crate::Error::Driver)
    }

    /// Write into the buffer. Waits for any in-flight GPU writes to finish
    /// before returning the slice.
    pub fn as_slice_mut(&mut self) -> crate::Result<&mut [T]> {
        self.inner.as_mut_slice().map_err(crate::Error::Driver)
    }

    /// Copy `data` into the buffer. `data.len()` must equal `self.len()`.
    pub fn fill_from(&mut self, data: &[T]) -> crate::Result<()> {
        let dst = self.as_slice_mut()?;
        assert_eq!(
            data.len(),
            dst.len(),
            "fill_from: length mismatch ({} vs {})",
            data.len(),
            dst.len()
        );
        // ptr::copy_nonoverlapping is a plain memcpy — no Copy bound needed.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst.as_mut_ptr(), data.len()) };
        Ok(())
    }
}

// allocate helpers used by GpuContext
pub(crate) unsafe fn alloc_raw<T: DeviceRepr + ValidAsZeroBits>(
    ctx: &std::sync::Arc<CudaContext>,
    len: usize,
) -> Result<PinnedBuffer<T>> {
    let inner = unsafe { ctx.alloc_pinned::<T>(len) }.map_err(crate::Error::Driver)?;
    Ok(PinnedBuffer { inner })
}
