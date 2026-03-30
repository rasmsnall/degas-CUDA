# degas-cuda

Run CUDA kernels from Rust. Handles device setup, typed GPU buffers, PTX loading, and kernel launching on top of [cudarc](https://github.com/coreylowman/cudarc).

Tested on: RTX 4070 Ti, driver 591.59, CUDA 13.1, Windows 11.

---

## Requirements

- An NVIDIA GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed (for `nvcc` and the driver headers)
- Rust 1.88 or newer

---

## Using it in your project

Add it as a git dependency in your `Cargo.toml`:

```toml
[dependencies]
degas-cuda = { git = "https://github.com/rasmsnall/degas-CUDA", features = ["cuda-13010"] }
```

**You must pick a `cuda-XXXXX` feature that matches your driver.** Run `nvidia-smi` and look at the top-right corner:

```
| Driver Version: 591.59    CUDA Version: 13.1 |
                                         ^^^^
                                     use this number
```

CUDA 13.1 → `cuda-13010`. CUDA 12.4 → `cuda-12040`. And so on. If you use a version newer than what your driver supports, you'll get a panic on startup about a missing symbol.

Available features: `cuda-11040`, `cuda-11050`, `cuda-11060`, `cuda-11070`, `cuda-11080`, `cuda-12000`, `cuda-12010`, `cuda-12020`, `cuda-12030`, `cuda-12040`, `cuda-12050`, `cuda-12060`, `cuda-12080`, `cuda-12090`, `cuda-13000`, `cuda-13010`, `cuda-13020`

If `nvcc` is in your PATH you can use `cuda-version-from-build-system` instead and it will detect the version automatically.

---

## Quick example

```rust
use degas_cuda::{GpuContext, LaunchConfig};

// Your kernel as a PTX string — compile it with nvcc or use include_str!
const PTX: &str = r#"
.version 6.0
.target sm_30
.address_size 64
// ... your kernel here
"#;

fn main() -> degas_cuda::Result<()> {
    let ctx = GpuContext::new(0)?;
    println!("{}", ctx.info()?);

    let n = 1024_i32;
    let a = ctx.upload(&vec![1.0_f32; n as usize])?;
    let b = ctx.upload(&vec![2.0_f32; n as usize])?;
    let mut c = ctx.alloc_zeros::<f32>(n as usize)?;

    let module = ctx.load_ptx_src(PTX)?;
    let kernel = module.get_kernel("my_kernel")?;

    let mut launch = ctx.prepare(&kernel);
    launch.arg_buf(&a).arg_buf(&b).arg_buf_mut(&mut c).arg_val(&n);
    unsafe { launch.execute(ctx.launch_config_1d(n as u32))? };

    ctx.synchronize()?;
    let result = ctx.download(&c)?;
    println!("{:?}", &result[..4]);
    Ok(())
}
```

---

## Running the examples

Clone the repo, then:

```sh
git clone https://github.com/rasmsnall/degas-CUDA
cd degas-CUDA
```

Print info about your GPU:

```sh
cargo run --example device_info --features cuda-13010
```

Run a 64K element vector addition:

```sh
cargo run --example vector_add --features cuda-13010
```

Replace `cuda-13010` with whichever version matches your driver.

---

## Configuration

You can change the defaults with a JSON config file:

```json
{
  "device_id": 0,
  "block_size": 256,
  "sync_on_launch": false
}
```

| Field | Default | Description |
|---|---|---|
| `device_id` | `0` | Which GPU to use |
| `block_size` | `256` | Threads per block for 1-D launches |
| `sync_on_launch` | `false` | Block after every kernel launch — turn on when chasing bugs |

Load it with:

```rust
let config = GpuConfig::load_or_default("gpu_config.json")?;
let ctx = GpuContext::with_config(config)?;
```

---

## License

MIT
