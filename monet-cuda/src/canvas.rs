use degas_cuda::{DeviceRepr, GpuBuffer2d};

/// A 2-D typed buffer on the GPU.
///
/// Rows run along the Y axis, columns along the X axis. The data is laid out
/// in row-major order: element `(row, col)` is at flat index `row * width + col`.
///
/// Create a `Canvas` through [`Session::canvas_from`][crate::Session::canvas_from]
/// or [`Session::canvas_zeros`][crate::Session::canvas_zeros].
///
/// To run a kernel over a `Canvas`, pass the flat buffer and dimensions separately:
///
/// ```no_run
/// use monet_cuda::Session;
/// use degas_cuda::LaunchConfig;
///
/// fn example(gpu: &Session, module: &degas_cuda::GpuModule) -> monet_cuda::Result<()> {
///     let mut canvas = gpu.canvas_zeros::<f32>(64, 64)?;
///     let kernel = module.get_kernel("my_kernel")?;
///
///     let w = canvas.width() as i32;
///     let h = canvas.height() as i32;
///     let mut launch = gpu.context().prepare(&kernel);
///     launch
///         .arg_buf_mut(canvas.as_buffer_mut().as_flat_mut())
///         .arg_val(&w)
///         .arg_val(&h);
///     unsafe { launch.execute(gpu.context().launch_config_2d(w as u32, h as u32))? };
///     Ok(())
/// }
/// ```
pub struct Canvas<T: DeviceRepr> {
    pub(crate) buf: GpuBuffer2d<T>,
}

impl<T: DeviceRepr> Canvas<T> {
    /// Number of columns.
    #[inline]
    pub fn width(&self) -> usize {
        self.buf.width()
    }

    /// Number of rows.
    #[inline]
    pub fn height(&self) -> usize {
        self.buf.height()
    }

    /// Total number of elements (`width * height`).
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

    /// Read-only access to the underlying [`GpuBuffer2d`].
    #[inline]
    pub fn as_buffer(&self) -> &GpuBuffer2d<T> {
        &self.buf
    }

    /// Mutable access to the underlying [`GpuBuffer2d`].
    #[inline]
    pub fn as_buffer_mut(&mut self) -> &mut GpuBuffer2d<T> {
        &mut self.buf
    }
}
