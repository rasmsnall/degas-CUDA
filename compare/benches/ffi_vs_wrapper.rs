//! Overhead comparison: degas-cuda wrapper vs raw cudarc.
//!
//! Each benchmark group runs the same GPU operation twice, once through the
//! degas-cuda API and once by calling cudarc directly, so you can see exactly
//! how much the wrapper costs.
//!
//! Run with:
//!   cargo bench -p compare --features cuda-13010
//!
//! The HTML report lands in target/criterion/. Open index.html in a browser.
//!
//! What each group measures:
//!
//!   upload        - copy a host Vec<f32> to GPU memory
//!   download      - copy GPU memory back to a host Vec<f32>
//!   roundtrip     - upload + sync + download combined
//!   kernel_launch - queue a no-op kernel (measures launch overhead only)
//!
//! All timings include CUDA stream synchronisation so you're measuring wall
//! time the caller actually waits for, not just queue time.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

// A trivial identity kernel that reads one element and writes it back.
// Used to measure kernel launch overhead rather than computation time.
// Targets sm_52 (Maxwell), the oldest arch supported by CUDA 12+.

const IDENTITY_PTX: &str = r#"
.version 8.0
.target sm_52
.address_size 64

.visible .entry identity(
    .param .u64 buf,
    .param .s32 n
)
{
    .reg .pred  %p;
    .reg .b32   %r<5>;
    .reg .b64   %rd<3>;
    .reg .f32   %f;

    ld.param.u64    %rd0, [buf];
    ld.param.s32    %r0,  [n];

    mov.u32         %r1, %ctaid.x;
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %tid.x;
    mad.lo.u32      %r4, %r1, %r2, %r3;

    setp.ge.s32     %p, %r4, %r0;
    @%p bra         done;

    cvt.u64.u32     %rd1, %r4;
    shl.b64         %rd1, %rd1, 2;
    add.u64         %rd2, %rd0, %rd1;

    ld.global.f32   %f, [%rd2];
    st.global.f32   [%rd2], %f;

done:
    ret;
}
"#;

const SIZES: &[usize] = &[1_024, 65_536, 1_048_576, 16_777_216];

fn bench_upload(c: &mut Criterion) {
    let mut group = c.benchmark_group("upload");

    for &n in SIZES {
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // degas-cuda
        group.bench_with_input(BenchmarkId::new("degas", n), &n, |b, _| {
            let ctx = degas_cuda::GpuContext::new(0).unwrap();
            b.iter(|| {
                let buf = ctx.upload(black_box(&host)).unwrap();
                ctx.synchronize().unwrap();
                drop(buf);
            });
        });

        // raw cudarc
        group.bench_with_input(BenchmarkId::new("cudarc", n), &n, |b, _| {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            b.iter(|| {
                let buf = stream.clone_htod(black_box(&host)).unwrap();
                stream.synchronize().unwrap();
                drop(buf);
            });
        });
    }

    group.finish();
}

fn bench_download(c: &mut Criterion) {
    let mut group = c.benchmark_group("download");

    for &n in SIZES {
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // degas-cuda
        group.bench_with_input(BenchmarkId::new("degas", n), &n, |b, _| {
            let ctx = degas_cuda::GpuContext::new(0).unwrap();
            let buf = ctx.upload(&host).unwrap();
            ctx.synchronize().unwrap();
            b.iter(|| {
                let v = ctx.download(black_box(&buf)).unwrap();
                black_box(v);
            });
        });

        // raw cudarc
        group.bench_with_input(BenchmarkId::new("cudarc", n), &n, |b, _| {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            let buf = stream.clone_htod(&host).unwrap();
            stream.synchronize().unwrap();
            b.iter(|| {
                let v = stream.clone_dtoh(black_box(&buf)).unwrap();
                black_box(v);
            });
        });
    }

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");

    for &n in SIZES {
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // degas-cuda
        group.bench_with_input(BenchmarkId::new("degas", n), &n, |b, _| {
            let ctx = degas_cuda::GpuContext::new(0).unwrap();
            b.iter(|| {
                let buf = ctx.upload(black_box(&host)).unwrap();
                ctx.synchronize().unwrap();
                let v = ctx.download(&buf).unwrap();
                black_box(v);
            });
        });

        // raw cudarc
        group.bench_with_input(BenchmarkId::new("cudarc", n), &n, |b, _| {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            b.iter(|| {
                let buf = stream.clone_htod(black_box(&host)).unwrap();
                stream.synchronize().unwrap();
                let v = stream.clone_dtoh(&buf).unwrap();
                black_box(v);
            });
        });
    }

    group.finish();
}

// Both sides launch the same identity kernel over the same buffer.
// The only difference is how much setup code you write to get there.

fn bench_kernel_launch(c: &mut Criterion) {
    use cudarc::driver::LaunchConfig;

    let mut group = c.benchmark_group("kernel_launch");

    for &n in SIZES {
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // degas-cuda
        group.bench_with_input(BenchmarkId::new("degas", n), &n, |b, _| {
            let ctx = degas_cuda::GpuContext::new(0).unwrap();
            let mut buf = ctx.upload(&host).unwrap();
            ctx.synchronize().unwrap();
            let module = ctx.load_ptx_src(IDENTITY_PTX).unwrap();
            let kernel = module.get_kernel("identity").unwrap();
            let n_i32 = n as i32;
            b.iter(|| {
                let mut launch = ctx.prepare(&kernel);
                launch
                    .arg_buf_mut(black_box(&mut buf))
                    .arg_val(&n_i32);
                unsafe { launch.execute(ctx.launch_config_1d(n as u32)).unwrap() };
                ctx.synchronize().unwrap();
            });
        });

        // raw cudarc
        group.bench_with_input(BenchmarkId::new("cudarc", n), &n, |b, _| {
            use cudarc::driver::PushKernelArg;
            use cudarc::nvrtc::Ptx;

            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            let mut buf = stream.clone_htod(&host).unwrap();
            stream.synchronize().unwrap();
            let module = ctx.load_module(Ptx::from_src(IDENTITY_PTX)).unwrap();
            let func = module.load_function("identity").unwrap();
            let n_i32 = n as i32;
            let block = 256_u32;
            let grid = (n as u32).div_ceil(block);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            b.iter(|| {
                let mut args = stream.launch_builder(&func);
                args.arg(black_box(&mut buf));
                args.arg(&n_i32);
                unsafe { args.launch(cfg).unwrap() };
                stream.synchronize().unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_upload,
    bench_download,
    bench_roundtrip,
    bench_kernel_launch,
);
criterion_main!(benches);
