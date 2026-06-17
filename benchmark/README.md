# SOFIE Benchmark for Inference on Heterogeneous Architectures

Measures **inference latency and throughput** for ONNX models compiled by SOFIE and
executed via [Alpaka](https://github.com/alpaka-group/alpaka).  Optionally runs the
same models through **ONNX Runtime GPU** for a side-by-side comparison.

---

## Supported Backends

| Backend | CMake value | Status |
|---------|-------------|--------|
| NVIDIA CUDA | `CUDA` (default) | Supported |
| AMD HIP/ROCm | `HIP` | Planned — not yet implemented |

The target architecture is selected with `-DSOFIE_BENCHMARK_BACKEND=<value>` at
configure time.  Specifying any value other than `CUDA` is a **hard CMake error** until
the corresponding backend is implemented.

The generated inference code and timing harness are backend-agnostic: they use
`sofie_bench::AccTag`, `sofie_bench::Platform`, `sofie_bench::Queue`, and the
`SOFIE_BENCH_DEVICE_SYNC()` macro defined in `src/BenchmarkBackend.hxx`.  Only the
low-level toolkit (CUDA vs HIP) needs to be swapped to add a new backend.

---

## Quick Start

### 1. Add your models

```
benchmark/models/
  GNN_model.onnx
  simple_transformer.onnx
  resnet50.onnx
  ...
```

Re-run CMake after adding or removing files (it globs `models/*.onnx`).

### 2. Configure

```bash
# SOFIE inference only — CUDA backend (default)
cmake -B build -DSOFIE_BENCHMARK=ON /path/to/SOFIE

# Explicitly name the backend (useful for CI or future HIP support)
cmake -B build -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_BACKEND=CUDA /path/to/SOFIE

# With ONNX Runtime GPU comparison
cmake -B build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  /path/to/SOFIE

# Override the CUDA SM architecture (default: native GPU or sm_75)
cmake -B build -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_CUDA_ARCH="86" /path/to/SOFIE
```

| CMake flag | Default | Description |
|---|---|---|
| `-DSOFIE_BENCHMARK=ON` | — | Enable the benchmark suite |
| `-DSOFIE_BENCHMARK_BACKEND=<val>` | `CUDA` | Target accelerator backend |
| `-DSOFIE_BENCHMARK_CUDA_ARCH=<sm>` | native / `75` | CUDA SM architecture(s), e.g. `86` for RTX 30xx, `80` for A100 |
| `-DSOFIE_BENCHMARK_ORT=ON` | `OFF` | Also benchmark ONNX Runtime GPU |
| `-DONNXRUNTIME_ROOT=<path>` | — | Path for ORT headers/library |
| `-DSOFIE_BENCHMARK_PROFILE=ON` | `OFF` | Enable per-operator GPU profiling instead of throughput benchmarking (see [Profiling](#profiling)) |
| `-DSOFIE_BENCHMARK_LARGE=ON` | `OFF` | Build `sofie_benchmark_large` for cluster GPUs (A100/H100, ≥40 GB VRAM) |
| `-DSOFIE_BENCHMARK_LARGE_CUDA_ARCH=<sm>` | `80` | CUDA SM architecture for the large-input benchmark |

> **Tested with ONNX Runtime 1.22.0 GPU**
> (`onnxruntime-linux-x64-gpu-1.22.0`).  The CMake config bundled with some ORT
> installations may reference an incorrect `lib64/` path — this toolkit uses manual
> header/library detection to avoid that.

### 3. Build

```bash
cmake --build build --target sofie_benchmark -j$(nproc)
```

This automatically:
1. Builds **`sofie_benchmark_emitter`** — parses each `.onnx` and emits:
   - `<Model>_GPU_ALPAKA.hxx` — SOFIE Alpaka inference code
   - `<Model>_GPU_ALPAKA.dat` — serialized weights
   - `<Model>_bench.hxx`      — timing wrapper `Benchmark_<Model>()`
2. Builds **`sofie_benchmark`** — compiles all generated code and links the timing loop.

### 4. Run

```bash
cd build/benchmark

# SOFIE only (no ORT needed at runtime)
./sofie_benchmark

# SOFIE + ONNX Runtime GPU comparison
LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH \
./sofie_benchmark --onnxruntime
```

---

## Runtime Options

| Flag | Default | Description |
|------|---------|-------------|
| `--warmup,     -w <N>` | 10  | Warm-up iterations (not timed) |
| `--iterations, -n <N>` | 100 | Timed iterations |
| `--weights-dir <path>` | `.` | Directory containing `.dat` weight files |
| `--onnxruntime, --ort` | off | Run ONNX Runtime GPU benchmark after each SOFIE model |
| `--help,       -h`     |     | Print this help and exit |

---

## Large-input Benchmark (`sofie_benchmark_large`)

For cluster GPUs (A100/H100/MI300X with ≥40 GB VRAM) a separate target is available
that includes models excluded from the default benchmark due to memory constraints on
consumer cards (≤8 GB):

```bash
cmake -B build -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_LARGE=ON \
      -DSOFIE_BENCHMARK_LARGE_CUDA_ARCH=80   # 80=A100, 90=H100
cmake --build build --target sofie_benchmark_large -j$(nproc)
```

The large-benchmark binary links CUDA runtime statically so it can run on cluster
nodes where the CUDA toolkit is not installed system-wide.

---

## Profiling

Profiling and throughput benchmarking are **mutually exclusive** builds.  Rebuild
with `-DSOFIE_BENCHMARK_PROFILE=ON` to switch the binary into profiling mode: the
timed H2D/inference/D2H loops are replaced by a profiling pass that measures
per-operator GPU time and prints a CPU/GPU memory breakdown.  The target backend
and CUDA architecture are controlled by the same `SOFIE_BENCHMARK_BACKEND` and
`SOFIE_BENCHMARK_CUDA_ARCH` flags used for benchmarking.

```bash
cmake -B build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_PROFILE=ON \
  /path/to/SOFIE
cmake --build build --target sofie_benchmark -j$(nproc)
cd build/benchmark && ./sofie_benchmark
```

After the normal throughput table, each model will print two additional blocks:

**GPU Profiling Results** — per-operator wall-clock time (microseconds) measured
with `std::chrono` and an `alpaka::wait(queue)` synchronisation point after every
kernel.  Results are sorted by average time descending, with ± stderr over all
timed iterations.  Warmup iterations are excluded (the session is reset before the
timed runs start).

```
============================================================
           GPU PROFILING RESULTS
   (wall-clock with alpaka::wait synchronization)
============================================================
  MatMul_3                      : 142.718 +/- 0.412 us  (100 runs)
  MatMul_1                      : 138.005 +/- 0.389 us  (100 runs)
  LayerNorm_5                   :  23.441 +/- 0.201 us  (100 runs)
  ...
  Overall_Time                  : 847.332 +/- 1.104 us  (100 runs)
============================================================
```

**Memory Usage Breakdown** — sizes computed at code-generation time from tensor
shapes and types.  No runtime measurement is needed; the values are embedded
as constants in the generated session code.

```
============================================================
              MEMORY USAGE BREAKDOWN
============================================================
  CPU Memory:
    Constant/embedded tensors : 0 bytes  (0.0000 MB)
    Weight tensors            : 12582912 bytes  (12.000 MB)
    Intermediate memory pool  : 0 bytes  (0.0000 MB)
    Total CPU                 : 12582912 bytes  (12.000 MB)
  GPU Memory (device buffers):
    Weight device buffers     : 12582912 bytes  (12.000 MB)
    Intermediate device bufs  : 4194304 bytes  (4.000 MB)
    Total GPU                 : 16777216 bytes  (16.000 MB)
============================================================
```

> **Note:** Profiling and benchmarking are mutually exclusive.  In a profiling
> build the throughput table is not printed; in a benchmark build
> `PrintProfilingResults` / `PrintMemoryInfo` are not called.  Rebuild without
> `-DSOFIE_BENCHMARK_PROFILE=ON` to measure peak throughput.

The same flag works for the large-input benchmark:

```bash
cmake -B build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_LARGE=ON \
  -DSOFIE_BENCHMARK_PROFILE=ON \
  /path/to/SOFIE
cmake --build build --target sofie_benchmark_large -j$(nproc)
```

---

## Re-running after adding models

```bash
cmake build
cmake --build build --target sofie_benchmark -j$(nproc)
```

---

## Adding a New Backend (HIP/ROCm)

The benchmark infrastructure is designed so adding a new backend requires changes
in only a few places:

1. **`CMakeLists.txt`** — add `"HIP"` to the `SOFIE_BENCHMARK_BACKEND` allowed
   values, call `enable_language(HIP)`, find `hip::host`, and set
   `_SOFIE_BENCH_ALPAKA_DEFINE = ALPAKA_ACC_GPU_HIP_ENABLED` /
   `_SOFIE_BENCH_BACKEND_DEFINE = SOFIE_BACKEND_HIP`.
2. **`src/BenchmarkBackend.hxx`** — already contains the `SOFIE_BACKEND_HIP` branch
   with `alpaka::TagGpuHipRt` aliases and `hipDeviceSynchronize()` sync macro.
3. **`src/ModelBench.cu.in`** — rename to `.hip.in` (or use a common extension) and
   configure the source file language property to `HIP`.
4. **`src/ONNXRuntimeBenchmark.hxx`** — swap `OrtCUDAProviderOptions` for the ROCm
   execution provider options if ORT comparison is desired on AMD hardware.
