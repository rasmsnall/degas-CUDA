//! # degas-cuda
//!
//! Run CUDA kernels from Rust without fighting the CUDA driver API directly.
//! You get device setup, typed GPU buffers, PTX loading, and kernel launching.
//! Most things are safe — the only `unsafe` is the actual kernel call, because
//! there's no way to verify argument types at compile time.
//!
//! Everything is built on top of [`cudarc`](https://docs.rs/cudarc).
//!
//! ## Getting started
//!
//! ```no_run
//! use degas_cuda::{GpuContext, LaunchConfig};
//!
//! // Load your PTX — compile it with nvcc or embed it with include_str!
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
//! ## Feature flags
//!
//! | Flag | What it does |
//! |------|-------------|
//! | `nvrtc-compile` | Exposes `compile_ptx` so you can compile CUDA C strings at runtime. You need the NVRTC shared library installed for this to work. |

mod buffer;
mod config;
mod context;
mod error;
mod info;
mod kernel;
mod module;

pub use buffer::GpuBuffer;
pub use config::GpuConfig;
pub use context::GpuContext;
pub use error::{Error, Result};
pub use info::DeviceInfo;
pub use kernel::{GpuKernel, KernelLaunch};
pub use module::GpuModule;

/// Grid and block dimensions for a kernel launch. Re-exported from `cudarc`.
pub use cudarc::driver::LaunchConfig;

/// Marker trait for types that can live in GPU memory or be passed as kernel
/// arguments. Re-exported from `cudarc` — implement this for your own
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
