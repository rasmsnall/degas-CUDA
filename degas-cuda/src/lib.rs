//! # degas-cuda
//!
//! Run CUDA kernels from Rust without fighting the CUDA driver API directly.
//! You get device setup, typed GPU buffers, PTX loading, and kernel launching.
//! Most things are safe; the only `unsafe` is the actual kernel call, because
//! there's no way to verify argument types at compile time.
//!
//! Everything is built on top of [`cudarc`](https://docs.rs/cudarc).
//!
//! ## Getting started
//!
//! ```no_run
//! use degas_cuda::{GpuContext, LaunchConfig};
//!
//! // Load your PTX. Compile it with nvcc or embed it with include_str!
//! const PTX: &str = include_str!("../kernels/scale.ptx");
//!
//! fn main() -> degas_cuda::Result<()> {
//!     let ctx = GpuContext::new(0)?;      // device 0
//!     println!("{}", ctx.info()?);
//!
//!     let n = 1024_i32;
//!     let mut buf = ctx.upload(&vec![1.0_f32; n as usize])?;
//!
//!     let module = ctx.load_ptx_src(PTX)?;
//!     let kernel = module.get_kernel("scale")?;
//!
//!     let factor = 2.0_f32;
//!     let mut launch = ctx.prepare(&kernel);
//!     launch.arg_buf_mut(&mut buf).arg_val(&factor).arg_val(&n);
//!     unsafe { launch.execute(LaunchConfig::for_num_elems(n as u32))? };
//!
//!     ctx.synchronize()?;
//!     let result = ctx.download(&buf)?;
//!     assert_eq!(result[0], 2.0);
//!     Ok(())
//! }
//! ```
//!
//! ## Picking the right CUDA version feature
//!
//! cudarc needs to know which CUDA version to build against. The version
//! must match what your driver supports, not the CUDA toolkit version.
//! These are often different; the toolkit can be newer than the driver.
//!
//! Run `nvidia-smi` and read the "CUDA Version" field in the top-right corner:
//!
//! ```text
//! +---------------------------+
//! | Driver Version: 591.59    |
//! | CUDA Version: 13.1        |  <- use this number
//! +---------------------------+
//! ```
//!
//! Then set the matching feature in your `Cargo.toml`. For CUDA 13.1 that's
//! `cuda-13010` (major=13, minor=01, patch=0):
//!
//! ```toml
//! [dependencies]
//! degas-cuda = { version = "0.1", features = ["cuda-13010"] }
//! ```
//!
//! Available version features (from cudarc 0.19):
//! `cuda-11040` through `cuda-13020`. If you use a version higher than what
//! your driver supports, you'll get a runtime panic on startup complaining
//! about a missing symbol in `nvcuda.dll` / `libcuda.so`. When that happens,
//! lower the version until it matches `nvidia-smi`.
//!
//! ## Using `PushKernelArg` directly
//!
//! If you go through [`KernelLaunch::raw_args`] to build a launch manually,
//! you need `cudarc::driver::PushKernelArg` in scope to call `.arg()` on the
//! underlying `LaunchArgs`. The `arg_buf` / `arg_buf_mut` / `arg_val` methods
//! on [`KernelLaunch`] already handle this, so it only comes up if you're
//! dropping down to `raw_args`.
//!
//! ## Feature flags
//!
//! | Flag | What it does |
//! |------|-------------|
//! | `nvrtc-compile` | Exposes `compile_ptx` so you can compile CUDA C strings at runtime. You need the NVRTC shared library installed for this to work. |

mod buffer;
mod buffer2d;
mod buffer3d;
mod config;
mod context;
mod error;
mod info;
mod kernel;
mod module;
mod pinned;
mod stream;

#[cfg(feature = "async")]
mod async_ctx;

pub use buffer::GpuBuffer;
pub use buffer2d::GpuBuffer2d;
pub use buffer3d::GpuBuffer3d;
pub use config::GpuConfig;
pub use context::GpuContext;
pub use error::{Error, Result};
pub use info::DeviceInfo;
pub use kernel::{GpuKernel, KernelLaunch};
pub use module::GpuModule;
pub use pinned::PinnedBuffer;
pub use stream::GpuStream;

#[cfg(feature = "async")]
pub use async_ctx::AsyncGpuContext;

/// Re-exported so downstream crates (e.g. monet-cuda) can write `T: ValidAsZeroBits`
/// bounds without adding a direct cudarc dependency.
pub use cudarc::driver::ValidAsZeroBits;

/// Grid and block dimensions for a kernel launch. Re-exported from `cudarc`.
pub use cudarc::driver::LaunchConfig;

// Compile-time check: GpuContext must be Send + Sync because the docs say so
// and users wrap it in Arc for multi-threaded use. This will fail to compile
// if a cudarc change ever removes those impls.
trait _AssertSendSync: Send + Sync {}
impl _AssertSendSync for GpuContext {}

/// Marker trait for types that can live in GPU memory or be passed as kernel
/// arguments. Re-exported from `cudarc`. Implement this for your own
/// `#[repr(C)]` structs if you need them on the GPU.
pub use cudarc::driver::DeviceRepr;

/// A PTX or CUBIN image ready to be loaded onto the GPU. Re-exported from
/// `cudarc`. Pass one to [`GpuContext::load_module`] if you need more control
/// than the `load_ptx_src` / `load_ptx_file` shortcuts give you.
pub use cudarc::nvrtc::Ptx;

/// Compile a CUDA C source string to PTX at runtime.
///
/// Needs the NVRTC library (`libnvrtc.so` on Linux, `nvrtc64.dll` on Windows)
/// at runtime. Enable the `nvrtc-compile` crate feature to get this.
///
/// ```no_run
/// # #[cfg(feature = "nvrtc-compile")]
/// # {
/// use degas_cuda::{compile_ptx, GpuContext};
///
/// let src = r#"
///     extern "C" __global__ void add_one(float* x, int n) {
///         int i = blockIdx.x * blockDim.x + threadIdx.x;
///         if (i < n) x[i] += 1.0f;
///     }
/// "#;
///
/// let ctx = GpuContext::new(0)?;
/// let ptx = compile_ptx(src)?;
/// let module = ctx.load_module(ptx)?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "nvrtc-compile")]
pub use cudarc::nvrtc::compile_ptx;
