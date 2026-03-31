//! Async wrappers for GPU work.
//!
//! CUDA calls block the calling thread until the driver accepts the work.
//! That's fine in a plain Rust program, but in a tokio application it ties up
//! a worker thread and can starve other tasks. The types here wrap the blocking
//! calls with `tokio::task::spawn_blocking` so they run on a dedicated thread
//! pool and your async code stays responsive.
//!
//! Enable this module with the `async` feature:
//!
//! ```toml
//! degas-cuda = { version = "0.1", features = ["async", "cuda-13010"] }
//! ```
//!
//! ## Use cases
//!
//! ### Web server with GPU acceleration
//!
//! A tokio-based HTTP server handles many requests concurrently. Without async
//! wrappers a single slow GPU upload would block the thread handling it, meaning
//! the server can't accept new connections while the transfer runs. With
//! `AsyncGpuContext` each upload is handed off to a blocking thread, so the
//! async runtime keeps serving other requests in the meantime.
//!
//! ```no_run
//! # #[cfg(feature = "async")]
//! # {
//! use degas_cuda::AsyncGpuContext;
//!
//! #[tokio::main]
//! async fn main() -> degas_cuda::Result<()> {
//!     let ctx = AsyncGpuContext::new(0).await?;
//!
//!     // AsyncGpuContext is Clone, so share it across tasks without Arc.
//!     let ctx_a = ctx.clone();
//!     let ctx_b = ctx.clone();
//!     let task_a = tokio::spawn(async move {
//!         ctx_a.upload(vec![1.0_f32; 65_536]).await
//!     });
//!     let task_b = tokio::spawn(async move {
//!         ctx_b.upload(vec![2.0_f32; 65_536]).await
//!     });
//!     let (buf_a, buf_b) = tokio::try_join!(task_a, task_b)?;
//!     Ok(())
//! }
//! # }
//! ```
//!
//! ### Async pipeline: transfer and process overlap
//!
//! Upload the next batch while the GPU crunches the previous one. The uploads
//! live inside `spawn_blocking` tasks so they don't block the async executor,
//! and you can `.await` them at the point where you actually need the buffers.
//!
//! ```no_run
//! # #[cfg(feature = "async")]
//! # {
//! use degas_cuda::AsyncGpuContext;
//!
//! async fn pipeline(ctx: &AsyncGpuContext, batches: &[Vec<f32>]) -> degas_cuda::Result<()> {
//!     for batch in batches {
//!         // upload takes ownership; clone the batch so the caller keeps it.
//!         let buf = ctx.upload(batch.clone()).await?;
//!         // kernel launch, download, etc.
//!     }
//!     Ok(())
//! }
//! # }
//! ```
//!
//! ### Concurrent GPU tasks with a timeout
//!
//! `tokio::time::timeout` lets you put a deadline on any async future, including
//! GPU work. If a kernel hangs longer than expected you'll get an error instead
//! of a frozen process.
//!
//! ```no_run
//! # #[cfg(feature = "async")]
//! # {
//! use std::time::Duration;
//! use degas_cuda::AsyncGpuContext;
//!
//! async fn guarded_upload(ctx: &AsyncGpuContext, data: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
//!     tokio::time::timeout(Duration::from_secs(5), ctx.upload(data))
//!         .await
//!         .map_err(|_| "GPU upload timed out")?
//!         .map(|_| ())
//!         .map_err(Into::into)
//! }
//! # }
//! ```
//!
//! ### Running arbitrary blocking GPU work
//!
//! Use [`AsyncGpuContext::run`] when none of the built-in helpers cover what
//! you need. Pass a closure that receives a reference to the inner `GpuContext`
//! and does whatever blocking work you like.
//!
//! ```no_run
//! # #[cfg(feature = "async")]
//! # {
//! use std::sync::Arc;
//! use degas_cuda::{AsyncGpuContext, GpuContext, LaunchConfig};
//!
//! async fn launch_kernel(ctx: Arc<AsyncGpuContext>) -> degas_cuda::Result<()> {
//!     ctx.run(|gpu| {
//!         let module = gpu.load_ptx_src(include_str!("../kernels/scale.ptx"))?;
//!         let kernel = module.get_kernel("scale")?;
//!         let mut buf = gpu.upload(&vec![1.0_f32; 1024])?;
//!         let factor = 2.0_f32;
//!         let n = 1024_i32;
//!         let mut launch = gpu.prepare(&kernel);
//!         launch.arg_buf_mut(&mut buf).arg_val(&factor).arg_val(&n);
//!         unsafe { launch.execute(LaunchConfig::for_num_elems(n as u32))? };
//!         gpu.synchronize()
//!     })
//!     .await
//! }
//! # }
//! ```

use std::sync::Arc;

use crate::{GpuBuffer, GpuContext, Result};
use cudarc::driver::DeviceRepr;

/// An async-friendly handle to a GPU device.
///
/// Cloning is cheap: it just increments the reference count on the inner `Arc`.
///
/// Each method submits a blocking CUDA call to `tokio::task::spawn_blocking`
/// so the tokio runtime is never blocked. Construct one with [`new`][Self::new]
/// or wrap an existing [`GpuContext`] with [`with_context`][Self::with_context].
///
/// The context is stored in an `Arc` internally. You can clone the `Arc` and
/// share `AsyncGpuContext` across tasks freely; CUDA itself serialises concurrent
/// access to the same device.
pub struct AsyncGpuContext {
    inner: Arc<GpuContext>,
}

impl Clone for AsyncGpuContext {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl AsyncGpuContext {
    /// Open device `device_id`. `0` selects the first GPU.
    pub async fn new(device_id: usize) -> Result<Self> {
        tokio::task::spawn_blocking(move || GpuContext::new(device_id))
            .await
            .map_err(|e| crate::Error::JoinError(e.to_string()))
            .and_then(|r| r.map(|ctx| Self { inner: Arc::new(ctx) }))
    }

    /// Wrap an existing `GpuContext`. Useful when you built the context
    /// elsewhere (e.g., with `GpuConfig`) and want async wrappers around it.
    pub fn with_context(ctx: GpuContext) -> Self {
        Self { inner: Arc::new(ctx) }
    }

    /// The underlying [`GpuContext`] for operations not covered here.
    pub fn context(&self) -> &GpuContext {
        &self.inner
    }

    /// Run an arbitrary blocking closure on the GPU context.
    ///
    /// The closure receives a reference to the inner `GpuContext` and can do
    /// anything a normal blocking call can do: load PTX, launch kernels, etc.
    /// The closure is executed on tokio's blocking thread pool so the async
    /// runtime stays responsive.
    ///
    /// The closure must be `'static`, meaning it cannot borrow from the calling
    /// scope. Move any data you need into the closure with `move`:
    ///
    /// ```no_run
    /// # #[cfg(feature = "async")]
    /// # {
    /// # use degas_cuda::AsyncGpuContext;
    /// # async fn example(ctx: AsyncGpuContext) -> degas_cuda::Result<()> {
    /// let data = vec![1.0_f32; 1024]; // owned, not borrowed
    /// ctx.run(move |gpu| {
    ///     let buf = gpu.upload(&data)?;  // data moved in
    ///     gpu.synchronize()
    /// }).await
    /// # }
    /// # }
    /// ```
    pub async fn run<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&GpuContext) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let ctx = self.inner.clone();
        tokio::task::spawn_blocking(move || f(&ctx))
            .await
            .map_err(|e| crate::Error::JoinError(e.to_string()))
            .and_then(|r| r)
    }

    /// Copy data to GPU memory. Takes ownership of `data`; if you need the
    /// original after uploading, clone it before calling this.
    /// The buffer is ready to use once the future resolves.
    pub async fn upload<T>(&self, data: Vec<T>) -> Result<GpuBuffer<T>>
    where
        T: DeviceRepr + Send + 'static,
    {
        let ctx = self.inner.clone();
        tokio::task::spawn_blocking(move || ctx.upload(&data))
            .await
            .map_err(|e| crate::Error::JoinError(e.to_string()))
            .and_then(|r| r)
    }

    /// Copy a GPU buffer back to the CPU as a `Vec`. Blocks the backing thread
    /// until the transfer completes, but does not block the tokio runtime.
    pub async fn download<T>(&self, buf: GpuBuffer<T>) -> Result<Vec<T>>
    where
        T: DeviceRepr + Send + 'static,
    {
        let ctx = self.inner.clone();
        tokio::task::spawn_blocking(move || ctx.download(&buf))
            .await
            .map_err(|e| crate::Error::JoinError(e.to_string()))
            .and_then(|r| r)
    }

    /// Block (on a backing thread) until all GPU work queued so far finishes.
    pub async fn synchronize(&self) -> Result<()> {
        let ctx = self.inner.clone();
        tokio::task::spawn_blocking(move || ctx.synchronize())
            .await
            .map_err(|e| crate::Error::JoinError(e.to_string()))
            .and_then(|r| r)
    }
}
