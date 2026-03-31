use cudarc::driver::DeviceRepr;

use crate::{Error, GpuBuffer, Result};

/// A typed 2-D block of GPU memory.
///
/// Internally this is a flat `GpuBuffer<T>` in row-major order, so element
/// `(row, col)` lives at index `row * width + col`. The GPU receives a plain
/// pointer; width and height are tracked on the Rust side so you can pass them
/// to kernels as separate arguments.
///
/// Create one with [`GpuContext::upload_2d`][crate::GpuContext::upload_2d]
/// or [`GpuContext::alloc_zeros_2d`][crate::GpuContext::alloc_zeros_2d].
/// GPU memory is freed when this value is dropped.
///
/// To pass the buffer to a kernel, call
/// [`as_flat`][GpuBuffer2d::as_flat] / [`as_flat_mut`][GpuBuffer2d::as_flat_mut]
/// to get the underlying `GpuBuffer<T>`, then use `arg_buf` / `arg_buf_mut`
/// as normal. Pass `width` and `height` as separate `arg_val` calls so the
/// kernel can compute its 2-D index.
pub struct GpuBuffer2d<T: DeviceRepr> {
    pub(crate) inner: GpuBuffer<T>,
    width: usize,
    height: usize,
}

impl<T: DeviceRepr> GpuBuffer2d<T> {
    /// Wrap an existing flat buffer with 2-D dimensions.
    ///
    /// Returns [`Error::DimensionOverflow`] if `width * height` overflows `usize`,
    /// or [`Error::SizeMismatch`] if `inner.len() != width * height`.
    pub fn from_buffer(inner: GpuBuffer<T>, width: usize, height: usize) -> Result<Self> {
        let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow)?;
        if inner.len() != expected {
            return Err(Error::SizeMismatch {
                src: inner.len(),
                dst: expected,
            });
        }
        Ok(Self { inner, width, height })
    }

    /// Number of columns.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of rows.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Total number of elements (`width * height`).
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Total size in bytes.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.inner.num_bytes()
    }

    /// Read-only access to the underlying flat buffer, for passing to kernels.
    #[inline]
    pub fn as_flat(&self) -> &GpuBuffer<T> {
        &self.inner
    }

    /// Mutable access to the underlying flat buffer, for passing to kernels
    /// that write to it.
    #[inline]
    pub fn as_flat_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.inner
    }

    /// Flat row-major index for a given `(row, col)` pair.
    /// No bounds checking; the kernel is responsible for staying in range.
    #[inline]
    pub fn flat_index(&self, row: usize, col: usize) -> usize {
        row * self.width + col
    }
}
