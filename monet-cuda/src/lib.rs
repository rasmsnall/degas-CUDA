//! # monet-cuda
//!
//! Higher-level GPU abstractions built on top of [`degas-cuda`].
//!
//! If `degas-cuda` is the engine, `monet-cuda` is the steering wheel. You get
//! shaped buffer types ([`Canvas`] for 2-D, [`Volume`] for 3-D, [`GpuVec`] for
//! 1-D) and a [`Session`] that cuts out the boilerplate of opening a device and
//! managing a stream. For anything lower-level, like custom PTX, raw kernel args,
//! or multi-stream pipelines, call [`Session::context`] and drop down to
//! `degas_cuda` directly.
//!
//! ## Quick example
//!
//! ```no_run
//! use monet_cuda::Session;
//!
//! fn main() -> monet_cuda::Result<()> {
//!     let gpu = Session::new(0)?;
//!     println!("{}", gpu.info()?);
//!
//!     // 1-D
//!     let vec = gpu.vec_from(&[1.0_f32, 2.0, 3.0])?;
//!     let back = gpu.download_vec(&vec)?;
//!     assert_eq!(back, &[1.0, 2.0, 3.0]);
//!
//!     // 2-D
//!     let data: Vec<f32> = (0..16 * 16).map(|i| i as f32).collect();
//!     let canvas = gpu.canvas_from(&data, 16, 16)?;
//!     println!("canvas {}x{}", canvas.width(), canvas.height());
//!
//!     // 3-D
//!     let vol = gpu.volume_zeros::<f32>(8, 8, 4)?;
//!     println!("volume {}x{}x{}", vol.width(), vol.height(), vol.depth());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Running custom kernels
//!
//! monet-cuda doesn't hide kernel launches; it just gives you shaped data
//! types. To run a kernel, grab the underlying `degas_cuda` types:
//!
//! ```no_run
//! use monet_cuda::Session;
//! use degas_cuda::LaunchConfig;
//!
//! fn run(gpu: &Session) -> monet_cuda::Result<()> {
//!     let mut canvas = gpu.canvas_zeros::<f32>(256, 256)?;
//!
//!     let module = gpu.context().load_ptx_src(include_str!("../kernels/fill.ptx"))?;
//!     let kernel = module.get_kernel("fill_f32")?;
//!
//!     let w = canvas.width() as i32;
//!     let h = canvas.height() as i32;
//!     let value = 1.0_f32;
//!
//!     let mut launch = gpu.context().prepare(&kernel);
//!     launch
//!         .arg_buf_mut(canvas.as_buffer_mut().as_flat_mut())
//!         .arg_val(&value)
//!         .arg_val(&w)
//!         .arg_val(&h);
//!
//!     unsafe {
//!         launch.execute(gpu.context().launch_config_2d(w as u32, h as u32))?;
//!     }
//!     gpu.context().synchronize()?;
//!     Ok(())
//! }
//! ```

mod canvas;
mod session;
mod vec;
mod volume;

pub use canvas::Canvas;
pub use session::Session;
pub use vec::GpuVec;
pub use volume::Volume;

// Re-export the error types so users only need to import monet-cuda.
pub use degas_cuda::{Error, Result};
