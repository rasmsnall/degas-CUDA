use degas_cuda::{DeviceRepr, GpuBuffer3d};

/// A 3-D typed buffer on the GPU.
///
/// Laid out as depth slices of rows: element `(d, row, col)` is at flat index
/// `(d * height + row) * width + col`. The GPU sees a plain pointer; the shape
/// is tracked on the Rust side so you can pass `width`, `height`, and `depth`
/// as kernel arguments.
///
/// Create a `Volume` through [`Session::volume_from`][crate::Session::volume_from]
/// or [`Session::volume_zeros`][crate::Session::volume_zeros].
///
/// To run a kernel over a `Volume`, pass the flat buffer and dimensions separately:
///
/// ```no_run
/// use monet_cuda::Session;
///
/// fn example(gpu: &Session, module: &degas_cuda::GpuModule) -> monet_cuda::Result<()> {
///     let mut vol = gpu.volume_zeros::<f32>(32, 32, 16)?;
///     let kernel = module.get_kernel("my_3d_kernel")?;
///
///     let w = vol.width() as i32;
///     let h = vol.height() as i32;
///     let d = vol.depth() as i32;
///     let mut launch = gpu.context().prepare(&kernel);
///     launch
///         .arg_buf_mut(vol.as_buffer_mut().as_flat_mut())
///         .arg_val(&w)
///         .arg_val(&h)
///         .arg_val(&d);
///     unsafe {
///         launch.execute(gpu.context().launch_config_3d(w as u32, h as u32, d as u32))?;
///     }
///     Ok(())
/// }
/// ```
pub struct Volume<T: DeviceRepr> {
    pub(crate) buf: GpuBuffer3d<T>,
}

impl<T: DeviceRepr> Volume<T> {
    /// Size along the X axis (columns).
    #[inline]
    pub fn width(&self) -> usize {
        self.buf.width()
    }

    /// Size along the Y axis (rows).
    #[inline]
    pub fn height(&self) -> usize {
        self.buf.height()
    }

    /// Size along the Z axis (depth slices).
    #[inline]
    pub fn depth(&self) -> usize {
        self.buf.depth()
    }

    /// Total number of elements (`width * height * depth`).
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

    /// Read-only access to the underlying [`GpuBuffer3d`].
    #[inline]
    pub fn as_buffer(&self) -> &GpuBuffer3d<T> {
        &self.buf
    }

    /// Mutable access to the underlying [`GpuBuffer3d`].
    #[inline]
    pub fn as_buffer_mut(&mut self) -> &mut GpuBuffer3d<T> {
        &mut self.buf
    }
}
