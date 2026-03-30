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
}

/// Every fallible function in this crate returns this.
///
/// ```
/// fn example() -> degas_cuda::Result<()> {
///     Ok(())
/// }
/// ```
pub type Result<T, E = Error> = std::result::Result<T, E>;
