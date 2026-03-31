use degas_cuda::{DeviceRepr, GpuBuffer};

/// A 1-D typed buffer on the GPU.
///
/// The monet-cuda equivalent of `GpuBuffer<T>`. Create one through
/// [`Session::vec_from`][crate::Session::vec_from] or
/// [`Session::vec_zeros`][crate::Session::vec_zeros].
///
/// To pass this to a kernel, use [`as_buffer`][GpuVec::as_buffer] or
/// [`as_buffer_mut`][GpuVec::as_buffer_mut] with
/// `arg_buf` / `arg_buf_mut` on a `KernelLaunch`.
pub struct GpuVec<T: DeviceRepr> {
    pub(crate) buf: GpuBuffer<T>,
}

impl<T: DeviceRepr> GpuVec<T> {
    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Total size in bytes.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.buf.num_bytes()
    }

    /// Read-only access to the underlying `GpuBuffer<T>` for kernel arguments.
    #[inline]
    pub fn as_buffer(&self) -> &GpuBuffer<T> {
        &self.buf
    }

    /// Mutable access to the underlying `GpuBuffer<T>` for kernel arguments
    /// that write to the buffer.
    #[inline]
    pub fn as_buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buf
    }
}
