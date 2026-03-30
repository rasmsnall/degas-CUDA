//! Enumerates all visible CUDA devices and prints their properties.
//!
//! Run with:
//!   cargo run --example device_info

use degas_cuda::GpuContext;

fn main() -> degas_cuda::Result<()> {
    let count = GpuContext::device_count()?;

    if count == 0 {
        eprintln!("No CUDA-capable devices found.");
        return Ok(());
    }

    println!("Found {count} CUDA device(s):\n");

    for i in 0..count {
        let ctx = GpuContext::new(i)?;
        let info = ctx.info()?;
        let (free, total) = ctx.raw().mem_get_info()?;

        println!("  [{i}] {}", info.name);
        println!("      Compute capability : {}.{}", info.compute_capability.0, info.compute_capability.1);
        println!("      Total memory       : {:.1} GiB", info.total_memory_bytes as f64 / (1 << 30) as f64);
        println!("      Free  memory       : {:.1} GiB", free as f64 / (1 << 30) as f64);
        println!("      Used  memory       : {:.1} GiB", (total - free) as f64 / (1 << 30) as f64);
        println!();
    }

    Ok(())
}
