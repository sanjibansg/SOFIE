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


## Nsight Compute Cross-Backend Profiling

Nsight Compute can profile SOFIE, ONNX Runtime GPU, and TensorRT using the same models. This mode captures the GPU kernels executed by exactly one steady-state inference for each model and backend.

This is separate from the internal SOFIE profiler described above. For Nsight Compute profiling, configure with:

```text
SOFIE_BENCHMARK_PROFILE=OFF
```

The runtime `--profile` flag starts the Nsight workflow. The benchmark performs one unprofiled priming inference, starts Nsight collection immediately before the measured inference, synchronizes the GPU, and then stops collection.

### 1. Configure and build

From the SOFIE repository root:

```bash
cmake -B benchmark/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DSOFIE_BENCHMARK_TRT=ON \
  -DSOFIE_BENCHMARK_PROFILE=OFF \
  -DSOFIE_BENCHMARK_CUDA_ARCH=<sm> \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  .
```

Replace `<sm>` with the CUDA architecture of the target GPU, for example `75`, `80`, `86`, or `100`.

Build the benchmark:

```bash
cmake --build benchmark/build --target sofie_benchmark -j$(nproc)
```

### 2. Generate the TensorRT engines first

TensorRT engine construction is slow and should not be performed inside Nsight Compute. It may fail or take an impractically long time because Nsight instruments and replays GPU kernels.

Run TensorRT once without profiling so that its serialized engine plans are generated and cached:

```bash
cd benchmark/build/benchmark

./sofie_benchmark --tensorrt
```

Wait for this run to finish before starting Nsight profiling.

The cached TensorRT plans will then be loaded by the profiling run instead of being rebuilt under Nsight.

### 3. Run Nsight Compute profiling

Nsight Compute may require administrator privileges to access GPU performance counters.

From `benchmark/build/benchmark`, run:

```bash
sudo ./sofie_benchmark \
  --profile \
  --onnxruntime \
  --tensorrt \
  --ncu /usr/local/cuda/bin/ncu
```

Use the actual path to `ncu` if it is installed elsewhere.

The profiler runs every model and backend in a separate subprocess. This gives each measurement a fresh CUDA context and prevents memory retained by one model or backend from affecting the next one.

The Nsight command uses:

```text
--profile-from-start off
```

so model initialization, backend initialization, TensorRT engine loading, and the priming inference are excluded from the captured statistics.

### 4. Profiling output

Each profiling run creates a timestamped results directory:

```text
benchmark/results/benchmark_<YYYYMMDD_HHMMSS>/
```

The directory contains separate results for each backend:

```text
benchmark_<timestamp>/
├── sofie/
│   ├── benchmark.csv
│   ├── <model>.ncu-rep
│   └── <model>.csv
├── ort/
│   ├── benchmark.csv
│   ├── <model>.ncu-rep
│   └── <model>.csv
└── tensorrt/
    ├── benchmark.csv
    ├── <model>.ncu-rep
    └── <model>.csv
```

The `.ncu-rep` files contain the complete Nsight Compute reports. The per-model `.csv` files contain the exported kernel metrics used by the summary script.

The conversion from `.ncu-rep` to `.csv` is performed near the end of the profiling workflow. Therefore, the per-model `.csv` files may not appear until all profiling subprocesses have completed.

### 5. Generate the profiling summary

Move to the directory containing the summary script:

```bash
cd benchmark/src
```

Then run:

```bash
sudo python3 summarize_profile.py \
  ../results/benchmark_<YYYYMMDD_HHMMSS>
```

Replace `<YYYYMMDD_HHMMSS>` with the timestamp of the profiling run.

The script generates:

```text
benchmark/results/benchmark_<timestamp>/profile_summary.md
benchmark/results/benchmark_<timestamp>/profile_summary.json
```

The Markdown report includes:

- kernel-launch counts;
- unique kernel-name counts;
- total GPU kernel time;
- average and maximum kernel duration;
- register usage;
- achieved occupancy;
- shared-memory usage;
- DRAM throughput and bandwidth;
- L1 and L2 cache hit rates;
- spilling detection;
- detected Tensor Core activity;
- likely memory-bound and compute-bound launches;
- per-model summaries;
- slowest kernels;
- highest-register kernels;
- lowest-occupancy kernels.

The JSON file contains the same results in a machine-readable format.

### 6. Important measurement note

Do not use the latency, throughput, transfer-time, or peak-memory values printed during an Nsight Compute run as normal benchmark results.

Nsight Compute instruments and may replay kernels several times to collect the requested sections. This significantly increases the measured wall-clock execution time.

Use separate runs for normal performance measurements and kernel-level profiling:

```bash
# Normal latency, throughput, transfers, and memory
./sofie_benchmark --onnxruntime --tensorrt
```

```bash
# Kernel-level Nsight Compute statistics
sudo ./sofie_benchmark \
  --profile \
  --onnxruntime \
  --tensorrt \
  --ncu /usr/local/cuda/bin/ncu
```

The Nsight results represent one captured steady-state inference per model and backend.

The reported total GPU time is the sum of the captured kernel durations and is not necessarily equal to end-to-end inference latency.


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
