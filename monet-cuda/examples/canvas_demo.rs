//! Demonstrates Session, Canvas, Volume, and GpuVec round-trips.
//!
//! Run with:
//!   cargo run -p monet-cuda --example canvas_demo --features cuda-13010

use monet_cuda::Session;

fn main() -> monet_cuda::Result<()> {
    let gpu = Session::new(0)?;
    println!("{}", gpu.info()?);

    // ── 1-D ──────────────────────────────────────────────────────────────────
    let data_1d: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let vec = gpu.vec_from(&data_1d)?;
    gpu.synchronize()?;
    let back = gpu.download_vec(&vec)?;
    assert_eq!(back, data_1d, "1-D round-trip mismatch");
    println!("vec OK — {} elements", back.len());

    // ── 2-D ──────────────────────────────────────────────────────────────────
    let width = 256_usize;
    let height = 128_usize;
    let data_2d: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    let canvas = gpu.canvas_from(&data_2d, width, height)?;
    gpu.synchronize()?;
    let back_2d = gpu.download_canvas(&canvas)?;
    assert_eq!(back_2d, data_2d, "2-D round-trip mismatch");
    println!(
        "canvas OK — {}×{} ({} elements)",
        canvas.width(),
        canvas.height(),
        back_2d.len()
    );

    // ── 3-D ──────────────────────────────────────────────────────────────────
    let vw = 32_usize;
    let vh = 32_usize;
    let vd = 16_usize;
    let data_3d: Vec<f32> = (0..vw * vh * vd).map(|i| i as f32).collect();

    let vol = gpu.volume_from(&data_3d, vw, vh, vd)?;
    gpu.synchronize()?;
    let back_3d = gpu.download_volume(&vol)?;
    assert_eq!(back_3d, data_3d, "3-D round-trip mismatch");
    println!(
        "volume OK — {}×{}×{} ({} elements)",
        vol.width(),
        vol.height(),
        vol.depth(),
        back_3d.len()
    );

    println!("canvas_demo passed.");
    Ok(())
}
