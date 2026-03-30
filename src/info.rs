/// Basic facts about a GPU. Get one from [`GpuContext::info`][crate::GpuContext::info].
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// The device index you passed to [`GpuContext::new`][crate::GpuContext::new].
    pub ordinal: usize,
    /// The device name, e.g. `"NVIDIA GeForce RTX 4090"`.
    pub name: String,
    /// Compute capability as `(major, minor)`, e.g. `(8, 9)` for Ada Lovelace.
    /// Higher is newer. Check NVIDIA's docs if you need to know what a given
    /// capability unlocks.
    pub compute_capability: (i32, i32),
    /// Total VRAM in bytes.
    pub total_memory_bytes: usize,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Device {} — {} (SM {}.{}, {:.1} GiB)",
            self.ordinal,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    }
}
