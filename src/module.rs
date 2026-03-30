use std::sync::Arc;

use cudarc::driver::CudaModule;

use crate::{Error, GpuKernel, Result};

/// A PTX module that's been loaded and JIT-compiled on the GPU.
///
/// Get one from [`GpuContext::load_ptx_src`][crate::GpuContext::load_ptx_src],
/// [`GpuContext::load_ptx_file`][crate::GpuContext::load_ptx_file], or
/// [`GpuContext::load_module`][crate::GpuContext::load_module].
///
/// A single module can hold multiple kernels — call [`get_kernel`][GpuModule::get_kernel]
/// for each one you need.
pub struct GpuModule {
    pub(crate) inner: Arc<CudaModule>,
}

impl GpuModule {
    /// Get a kernel by the name of its `__global__` function.
    ///
    /// If your PTX was compiled from CUDA C++ (not `extern "C"`), the name
    /// will be mangled — use `nvcc --ptx` and check the output to find the
    /// actual symbol name.
    ///
    /// ```no_run
    /// # let ctx = degas_cuda::GpuContext::new(0)?;
    /// # let module = ctx.load_ptx_src("")?;
    /// let kernel = module.get_kernel("vector_add")?;
    /// # Ok::<(), degas_cuda::Error>(())
    /// ```
    pub fn get_kernel(&self, name: &str) -> Result<GpuKernel> {
        self.inner
            .load_function(name)
            .map(|f| GpuKernel { inner: f })
            .map_err(|_| Error::KernelNotFound(name.to_owned()))
    }
}
