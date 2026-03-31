use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Result;

/// Settings for a [`GpuContext`][crate::GpuContext].
///
/// [`GpuContext::new`][crate::GpuContext::new] covers the common case, so
/// this is optional. Use it when you want to pick a specific GPU, tune the
/// block size, or turn on debug synchronisation.
///
/// Every field is optional in the JSON file. Anything you leave out just gets
/// the default value, so `{}` is a valid config.
///
/// # Loading from a file
///
/// ```no_run
/// use degas_cuda::{GpuConfig, GpuContext};
///
/// let config = GpuConfig::from_file("gpu_config.json")?;
/// let ctx = GpuContext::with_config(config)?;
/// # Ok::<(), degas_cuda::Error>(())
/// ```
///
/// # Building in code and saving
///
/// ```no_run
/// use degas_cuda::GpuConfig;
///
/// let config = GpuConfig {
///     device_id: 0,
///     block_size: 512,
///     sync_on_launch: true,
/// };
/// config.save("gpu_config.json")?;
/// # Ok::<(), degas_cuda::Error>(())
/// ```
///
/// # JSON format
///
/// ```json
/// {
///   "device_id": 0,
///   "block_size": 256,
///   "sync_on_launch": false
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// Which GPU to use. 0 is the first one. Call
    /// [`GpuContext::device_count`][crate::GpuContext::device_count] if you're
    /// not sure how many you have. Default: `0`.
    pub device_id: usize,

    /// Threads per block for 1-D launches via
    /// [`GpuContext::launch_config_1d`][crate::GpuContext::launch_config_1d].
    /// Keep it a multiple of 32 (warp size) and no higher than 1024.
    /// Default: `256`.
    pub block_size: u32,

    /// When true, every kernel launch blocks until the GPU finishes. That's
    /// slow, but it means any driver error you get points at the actual kernel
    /// that caused it rather than some random later call. Turn this on when
    /// you're chasing bugs, off in production. Default: `false`.
    pub sync_on_launch: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            block_size: 256,
            sync_on_launch: false,
        }
    }
}

impl GpuConfig {
    /// Read config from a JSON file.
    ///
    /// Fields you leave out fall back to their defaults, so you only need to
    /// include the ones you actually want to change.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&text)?;
        Ok(config)
    }

    /// Read config from a file if it exists, or return the defaults if it
    /// doesn't. Good for making the config file optional: your program works
    /// fine without one, and users can drop one in to override settings.
    ///
    /// Only returns an error if the file exists but is unreadable or broken.
    pub fn load_or_default(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.exists() {
            Self::from_file(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Write this config to a file as pretty-printed JSON.
    /// Creates the file if it doesn't exist, overwrites it if it does.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parse from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
