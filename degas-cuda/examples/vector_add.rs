//! Demonstrates element-wise f32 vector addition on the GPU.
//!
//! The PTX kernel is embedded directly in this file so no external `.ptx` file
//! or CUDA compiler is required.
//!
//! Run with:
//!   cargo run --example vector_add

use degas_cuda::{GpuContext, LaunchConfig};

/// Pre-compiled PTX for:
///
/// ```cuda
/// extern "C" __global__
/// void vector_add(const float* a, const float* b, float* c, int n) {
///     int i = blockIdx.x * blockDim.x + threadIdx.x;
///     if (i < n) c[i] = a[i] + b[i];
/// }
/// ```
///
/// Targets SM 3.0+ (Maxwell and newer). Recompile with `nvcc -ptx` for a
/// specific architecture if needed.
const VECTOR_ADD_PTX: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .s32 vector_add_param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<7>;

    ld.param.u64  %rd0, [vector_add_param_0];
    ld.param.u64  %rd1, [vector_add_param_1];
    ld.param.u64  %rd2, [vector_add_param_2];
    ld.param.s32  %r2,  [vector_add_param_3];

    mov.u32  %r3, %ctaid.x;
    mov.u32  %r4, %ntid.x;
    mov.u32  %r0, %tid.x;
    mad.lo.s32  %r1, %r3, %r4, %r0;

    setp.ge.s32  %p0, %r1, %r2;
    @%p0 bra  BB0_2;

    mul.wide.s32  %rd3, %r1, 4;
    add.u64  %rd4, %rd0, %rd3;
    add.u64  %rd5, %rd1, %rd3;
    add.u64  %rd6, %rd2, %rd3;

    ld.global.f32  %f0, [%rd4];
    ld.global.f32  %f1, [%rd5];
    add.f32        %f2, %f0, %f1;
    st.global.f32  [%rd6], %f2;

BB0_2:
    ret;
}
"#;

fn main() -> degas_cuda::Result<()> {
    const N: usize = 1024 * 64; // 64 K elements

    // ── Setup ────────────────────────────────────────────────────────────────
    let ctx = GpuContext::new(0)?;
    println!("{}", ctx.info()?);

    // ── Host data ────────────────────────────────────────────────────────────
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();

    // ── Upload to GPU ────────────────────────────────────────────────────────
    let a = ctx.upload(&a_host)?;
    let b = ctx.upload(&b_host)?;
    let mut c = ctx.alloc_zeros::<f32>(N)?;

    // ── Load kernel ──────────────────────────────────────────────────────────
    let module = ctx.load_ptx_src(VECTOR_ADD_PTX)?;
    let kernel = module.get_kernel("vector_add")?;

    // ── Launch ───────────────────────────────────────────────────────────────
    let n = N as i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut launch = ctx.prepare(&kernel);
    launch.arg_buf(&a).arg_buf(&b).arg_buf_mut(&mut c).arg_val(&n);

    // SAFETY: argument types and order match the PTX kernel signature above.
    unsafe { launch.execute(cfg)? };

    ctx.synchronize()?;

    // ── Verify ───────────────────────────────────────────────────────────────
    let c_host = ctx.download(&c)?;

    let errors = c_host
        .iter()
        .zip(a_host.iter().zip(b_host.iter()))
        .filter(|(c, (a, b))| (*c - (*a + *b)).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("vector_add OK — {N} elements verified.");
    } else {
        eprintln!("MISMATCH: {errors} wrong values out of {N}");
        std::process::exit(1);
    }

    Ok(())
}
