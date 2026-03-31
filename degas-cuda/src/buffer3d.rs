use cudarc::driver::DeviceRepr;

use crate::{Error, GpuBuffer, Result};

/// A typed 3-D block of GPU memory.
///
/// Laid out in depth-major, row-major order: element `(d, row, col)` lives
/// at flat index `(d * height + row) * width + col`. Like `GpuBuffer2d`, the
/// shape is tracked on the Rust side while the GPU receives a plain pointer.
///
/// Create one with [`GpuContext::upload_3d`][crate::GpuContext::upload_3d]
/// or [`GpuContext::alloc_zeros_3d`][crate::GpuContext::alloc_zeros_3d].
/// GPU memory is freed when this value is dropped.
///
/// To pass the buffer to a kernel, use
/// [`as_flat`][GpuBuffer3d::as_flat] / [`as_flat_mut`][GpuBuffer3d::as_flat_mut]
/// alongside `arg_buf` / `arg_buf_mut`, then pass `width`, `height`, and
/// `depth` as separate `arg_val` calls.
pub struct GpuBuffer3d<T: DeviceRepr> {
    pub(crate) inner: GpuBuffer<T>,
    width: usize,
    height: usize,
    depth: usize,
}

impl<T: DeviceRepr> GpuBuffer3d<T> {
    /// Wrap an existing flat buffer with 3-D dimensions.
    ///
    /// Returns [`Error::DimensionOverflow`] if `width * height * depth` overflows `usize`,
    /// or [`Error::SizeMismatch`] if `inner.len() != width * height * depth`.
    pub fn from_buffer(inner: GpuBuffer<T>, width: usize, height: usize, depth: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(depth))
            .ok_or(Error::DimensionOverflow)?;
        if inner.len() != expected {
            return Err(Error::SizeMismatch {
                src: inner.len(),
                dst: expected,
            });
        }
        Ok(Self { inner, width, height, depth })
    }

    /// Size along the X axis (columns).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Size along the Y axis (rows).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Size along the Z axis (slices).
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Total number of elements (`width * height * depth`).
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

    /// Flat index for a given `(depth, row, col)` triple.
    /// No bounds checking; the kernel is responsible for staying in range.
    #[inline]
    pub fn flat_index(&self, d: usize, row: usize, col: usize) -> usize {
        (d * self.height + row) * self.width + col
    }
}
