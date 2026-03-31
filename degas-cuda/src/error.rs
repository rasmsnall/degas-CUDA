use thiserror::Error;

/// Everything that can go wrong in this library.
#[derive(Debug, Error)]
pub enum Error {
    /// Something the CUDA driver rejected. The inner value carries the raw
    /// `CUresult` code and a description from cudarc.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// You asked for a kernel by name and it wasn't in the module. Check for
    /// typos and make sure the function is declared `extern "C"` if you
    /// compiled from CUDA C++.
    #[error("kernel `{0}` not found in module")]
    KernelNotFound(String),

    /// `upload_into` requires source and destination to be the same length.
    #[error("size mismatch: src has {src} elements but dst has {dst}")]
    SizeMismatch { src: usize, dst: usize },

    /// The config JSON was readable but couldn't be parsed.
    #[error("config parse error: {0}")]
    ConfigParse(#[from] serde_json::Error),

    /// A file read or write failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A `tokio::task::spawn_blocking` task panicked or was cancelled.
    /// Only produced by `AsyncGpuContext`. The inner string carries the
    /// panic message from the blocking thread.
    #[error("async task join error: {0}")]
    JoinError(String),

    /// A dimension calculation (e.g. `width * height`) would overflow `usize`.
    /// Reduce the requested dimensions.
    #[error("dimension calculation overflows usize")]
    DimensionOverflow,
}

/// Every fallible function in this crate returns this.
///
/// ```
/// fn example() -> degas_cuda::Result<()> {
///     Ok(())
/// }
/// ```
pub type Result<T, E = Error> = std::result::Result<T, E>;
