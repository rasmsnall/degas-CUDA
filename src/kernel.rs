use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg};

use crate::Result;

/// A handle to a `__global__` kernel function.
///
/// Get one from [`GpuModule::get_kernel`][crate::GpuModule::get_kernel], then
/// use [`GpuContext::prepare`][crate::GpuContext::prepare] to start setting up
/// a launch. If you want to run it on a non-default stream, use [`on`][GpuKernel::on]
/// directly.
pub struct GpuKernel {
    pub(crate) inner: CudaFunction,
}

impl GpuKernel {
    /// Set up a launch on a specific stream.
    ///
    /// Set `sync_after` to `true` if you want this launch to block until
    /// the kernel finishes — handy for debugging. In normal use just call
    /// [`GpuContext::prepare`][crate::GpuContext::prepare], which picks this
    /// up from the config automatically.
    pub fn on<'a>(&'a self, stream: &'a Arc<CudaStream>, sync_after: bool) -> KernelLaunch<'a> {
        KernelLaunch {
            args: stream.launch_builder(&self.inner),
            stream,
            sync_after,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Builds up a kernel call before firing it.
///
/// Push arguments in the same order the kernel expects them, then call
/// [`execute`][KernelLaunch::execute]. That's it.
///
/// If `sync_on_launch` is set in your [`GpuConfig`][crate::GpuConfig],
/// `execute` will wait for the kernel to finish before returning — useful
/// when you're trying to pin down where an error is coming from.
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
///     launch
///         .arg_buf(&a)
///         .arg_buf(&b)
///         .arg_buf_mut(&mut c)
///         .arg_val(&n);
///     unsafe { launch.execute(ctx.launch_config_1d(n as u32))? };
///     Ok(())
/// }
/// ```
pub struct KernelLaunch<'a> {
    args: LaunchArgs<'a>,
    stream: &'a Arc<CudaStream>,
    sync_after: bool,
}

impl<'a> KernelLaunch<'a> {
    /// Pass a read-only GPU buffer as the next argument.
    pub fn arg_buf<T: DeviceRepr>(&mut self, buf: &'a crate::GpuBuffer<T>) -> &mut Self {
        self.args.arg(&buf.inner);
        self
    }

    /// Pass a read-write GPU buffer as the next argument.
    /// Use this when the kernel writes to the buffer.
    pub fn arg_buf_mut<T: DeviceRepr>(&mut self, buf: &'a mut crate::GpuBuffer<T>) -> &mut Self {
        self.args.arg(&mut buf.inner);
        self
    }

    /// Pass a scalar value as the next argument. Any primitive numeric type
    /// works here (`i32`, `f32`, `u64`, etc.).
    pub fn arg_val<T: DeviceRepr>(&mut self, val: &'a T) -> &mut Self {
        self.args.arg(val);
        self
    }

    /// Fire the kernel.
    ///
    /// The launch is asynchronous by default — it queues the work on the
    /// stream and returns immediately. If `sync_on_launch` is on in your
    /// config, it blocks until the GPU finishes before returning.
    ///
    /// # Safety
    ///
    /// You have to get the arguments right yourself:
    /// - Same order as the kernel signature.
    /// - Buffers large enough for the thread count you're launching.
    /// - No two kernels on different streams writing the same buffer at the
    ///   same time.
    ///
    /// There's no way to check this at compile time, which is why this is
    /// `unsafe`.
    pub unsafe fn execute(&mut self, cfg: LaunchConfig) -> Result<()> {
        unsafe { self.args.launch(cfg) }
            .map(|_| ())
            .map_err(crate::Error::Driver)?;

        if self.sync_after {
            self.stream.synchronize().map_err(crate::Error::Driver)?;
        }

        Ok(())
    }

    /// Direct access to the underlying [`LaunchArgs`] if you need something
    /// we don't expose, like recording timing events or doing a cooperative
    /// launch.
    ///
    /// Note: to call `.arg()` on the returned `LaunchArgs` you need
    /// `cudarc::driver::PushKernelArg` in scope — it's a trait method.
    pub fn raw_args(&mut self) -> &mut LaunchArgs<'a> {
        &mut self.args
    }
}
