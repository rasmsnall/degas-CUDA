use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, DeviceRepr};

/// A typed block of memory on the GPU.
///
/// Create one with [`GpuContext::upload`][crate::GpuContext::upload] (copies
/// from a host slice) or [`GpuContext::alloc_zeros`][crate::GpuContext::alloc_zeros]
/// (allocates zeroed memory without a copy). GPU memory is freed when this
/// value is dropped.
///
/// To pass a buffer to a kernel, use
/// [`arg_buf`][crate::KernelLaunch::arg_buf] (read-only) or
/// [`arg_buf_mut`][crate::KernelLaunch::arg_buf_mut] (read-write).
pub struct GpuBuffer<T: DeviceRepr> {
    pub(crate) inner: CudaSlice<T>,
}

impl<T: DeviceRepr> GpuBuffer<T> {
    /// Number of elements, not bytes. For bytes see [`num_bytes`][Self::num_bytes].
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.inner.num_bytes()
    }

    /// A read-only view into the buffer. Good for slicing out a sub-range
    /// and passing it to a kernel without copying.
    #[inline]
    pub fn as_view(&self) -> CudaView<'_, T> {
        self.inner.as_view()
    }

    /// A mutable view into the buffer.
    #[inline]
    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.inner.as_view_mut()
    }

    /// Allocates a new buffer on the same device and copies the data into it.
    /// This is a device-to-device copy, so it doesn't go through the CPU.
    pub fn try_clone(&self) -> crate::Result<Self> {
        let inner = self.inner.try_clone()?;
        Ok(Self { inner })
    }
}
