# SOFIE Benchmark for Inference on Heterogeneous Architectures

The benchmark toolkit measures **inference latency, throughput, transfer time, GPU memory usage, and GPU utilisation** for ONNX models compiled by SOFIE and executed through Alpaka.

The same models can optionally be benchmarked with **ONNX Runtime GPU**, **TensorRT**, and **PyTorch AOTInductor** for comparison.

---

## Supported Accelerator Backends

| Backend | CMake value | Status |
|---|---|---|
| NVIDIA CUDA | `CUDA` | Supported |
| AMD HIP/ROCm | `HIP` | Planned |

The target accelerator backend is selected with `-DSOFIE_BENCHMARK_BACKEND=<value>`.

The generated SOFIE inference code and timing harness use the backend abstractions defined in `src/BenchmarkBackend.hxx`.

---

## Prerequisites

The CUDA benchmark requires:

- CMake 3.18 or newer;
- a C++20 compiler;
- an NVIDIA CUDA toolkit and compatible driver;
- NVML for GPU-utilisation sampling.

The comparison backends are optional and only need to be installed when their corresponding CMake option is enabled.

---

## Benchmark Models

By default, ONNX models are read from:

```text
benchmark/models/
```

For example:

```text
benchmark/models/
├── GNN_model.onnx
├── simple_transformer.onnx
├── resnet50.onnx
└── ...
```

Re-run CMake after adding or removing ONNX files because model discovery happens at configure time.

A different model directory can be selected with:

```bash
-DSOFIE_BENCHMARK_MODEL_DIR=<path>
```

Relative paths are resolved relative to `benchmark/`:

```bash
-DSOFIE_BENCHMARK_MODEL_DIR=models
```

Absolute paths are also accepted:

```bash
-DSOFIE_BENCHMARK_MODEL_DIR=/data/sofie/models
```

When PyTorch AOTInductor is enabled, the corresponding AOT packages must be stored under an `aot_models/` directory inside the selected model directory:

```text
<model-dir>/
├── model_a.onnx
├── model_b.onnx
└── aot_models/
    ├── model_a.pt2
    └── model_b.pt2
```

The `.onnx` and `.pt2` files must use the same model basename.

---

## Optional Comparison Backends

### ONNX Runtime GPU

Enable ONNX Runtime with:

```bash
-DSOFIE_BENCHMARK_ORT=ON
```

If ONNX Runtime is installed in a non-system location, specify its installation root:

```bash
-DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

The directory should contain the ONNX Runtime headers and libraries, typically:

```text
/path/to/onnxruntime/
├── include/
└── lib/
```

If the runtime linker cannot locate `libonnxruntime.so`, add the library directory to `LD_LIBRARY_PATH` before running:

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

---

### TensorRT

Enable TensorRT with:

```bash
-DSOFIE_BENCHMARK_TRT=ON
```

A system TensorRT installation is detected automatically.

For a TensorRT installation in a custom location, use:

```bash
-DTENSORRT_ROOT=/path/to/TensorRT
```

The TensorRT root should contain its headers and libraries, typically:

```text
/path/to/TensorRT/
├── include/
└── lib/
```

For a TensorRT tar installation, also make its libraries visible at runtime:

```bash
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
```

TensorRT engines are generated on first use and cached for later runs.

---

### PyTorch AOTInductor

Enable the PyTorch AOTInductor comparison with:

```bash
-DSOFIE_BENCHMARK_AOT=ON
```

The benchmark links directly against the PyTorch C++ libraries and loads `.pt2` AOTInductor packages using `AOTIModelPackageLoader`.

Install a CUDA-enabled PyTorch build first and verify it:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CMake must be able to locate the PyTorch CMake package. The recommended configuration is:

```bash
-DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

Alternatively, `Torch_DIR` can be supplied directly if required.

The repository contains family-specific AOT export scripts:

```text
benchmark/models/export_aot_gnn.py
benchmark/models/export_aot_gnn_large.py
benchmark/models/export_aot_transformer.py
benchmark/models/export_aot_punet.py
benchmark/models/export_aot_mambav2.py
```

Run the exporters required for the models being benchmarked before running the AOT comparison.

For example:

```bash
python3 benchmark/models/export_aot_gnn.py
python3 benchmark/models/export_aot_gnn_large.py
python3 benchmark/models/export_aot_transformer.py
```

The PUNet exporter additionally requires the corresponding TICL repository:

```bash
python3 benchmark/models/export_aot_punet.py \
  --ticl-repo /path/to/TICL-GNN-Trackster-Linking
```

The resulting `.pt2` packages must be present in:

```text
<model-dir>/aot_models/
```

If a custom `SOFIE_BENCHMARK_MODEL_DIR` is used, ensure the generated `.pt2` packages are copied or generated under that directory's `aot_models/` subdirectory.

---

### Mamba AOTInductor Models

The Mamba models require additional setup because their AOTInductor packages contain the custom operator:

```text
mamba_bench::selective_scan_fwd
```

The benchmark registers this operator in `src/MambaAOTCustomOps.cxx` and forwards it to Mamba's CUDA `selective_scan_cuda` implementation.

Install Mamba with the CUDA selective-scan extension enabled:

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE \
python3 -m pip install mamba-ssm --no-build-isolation
```

If a local CUDA build must be forced:

```bash
MAMBA_FORCE_BUILD=TRUE \
MAMBA_KEEP_CUDA_BUILD=TRUE \
python3 -m pip install mamba-ssm --no-build-isolation
```

Verify that the CUDA extension is available:

```bash
python3 -c "import selective_scan_cuda; print(selective_scan_cuda.__file__)"
```

Generate the Mamba AOT package using:

```bash
python3 benchmark/models/export_aot_mambav2.py
```

Then configure the benchmark with both:

```text
-DSOFIE_BENCHMARK_AOT=ON
-DSOFIE_BENCHMARK_MAMBA_AOT=ON
```

CMake first tries to locate `selective_scan_cuda` automatically beside the PyTorch installation.

If automatic detection fails, specify the library explicitly:

```bash
-DMAMBA_SELECTIVE_SCAN_LIBRARY=/path/to/selective_scan_cuda.so
```

The exact installed path can be obtained with:

```bash
python3 -c "import selective_scan_cuda; print(selective_scan_cuda.__file__)"
```

`SOFIE_BENCHMARK_MAMBA_AOT=ON` requires `SOFIE_BENCHMARK_AOT=ON`.

Without Mamba AOT support enabled, `mambav2_*` models are skipped only for the PyTorch AOTInductor comparison.

---

## Configure

### SOFIE only

From the SOFIE repository root:

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_BACKEND=CUDA \
  .
```

### SOFIE + ONNX Runtime

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  .
```

### SOFIE + TensorRT

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_TRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  .
```

`TENSORRT_ROOT` can be omitted for a system installation.

### SOFIE + PyTorch AOTInductor

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_AOT=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  .
```

### All comparison backends

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_BACKEND=CUDA \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DSOFIE_BENCHMARK_TRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DSOFIE_BENCHMARK_AOT=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  .
```

To additionally benchmark Mamba with AOTInductor:

```bash
cmake -B benchmark/build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DSOFIE_BENCHMARK_TRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DSOFIE_BENCHMARK_AOT=ON \
  -DSOFIE_BENCHMARK_MAMBA_AOT=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  .
```

If required, append:

```bash
-DMAMBA_SELECTIVE_SCAN_LIBRARY=/path/to/selective_scan_cuda.so
```

### CMake Options

| CMake option | Default | Description |
|---|---|---|
| `SOFIE_BENCHMARK` | — | Enable the benchmark suite |
| `SOFIE_BENCHMARK_BACKEND` | `CUDA` | Accelerator backend |
| `SOFIE_BENCHMARK_MODEL_DIR` | `models` | ONNX model directory; relative to `benchmark/` or absolute |
| `SOFIE_BENCHMARK_CUDA_ARCH` | `CMAKE_CUDA_ARCHITECTURES` / `75` | CUDA architecture |
| `SOFIE_BENCHMARK_ORT` | `OFF` | Enable ONNX Runtime GPU comparison |
| `ONNXRUNTIME_ROOT` | empty | Optional ONNX Runtime installation root |
| `SOFIE_BENCHMARK_TRT` | `OFF` | Enable TensorRT comparison |
| `TENSORRT_ROOT` | empty | Optional TensorRT installation root |
| `SOFIE_BENCHMARK_AOT` | `OFF` | Enable PyTorch AOTInductor comparison |
| `CMAKE_PREFIX_PATH` | — | Can be used to locate the installed PyTorch CMake package |
| `SOFIE_BENCHMARK_MAMBA_AOT` | `OFF` | Enable Mamba custom-op support for AOTInductor |
| `MAMBA_SELECTIVE_SCAN_LIBRARY` | auto | Optional explicit path to `selective_scan_cuda` |
| `SOFIE_BENCHMARK_LOWRANK` | `OFF` | Enable SOFIE low-rank factorization |
| `SOFIE_BENCHMARK_LOWRANK_RATIO` | `0.5` | Low-rank factorization ratio |
| `SOFIE_BENCHMARK_PROFILE` | `OFF` | Enable internal SOFIE per-operator profiling instead of throughput benchmarking |
| `SOFIE_BENCHMARK_LARGE` | `OFF` | Build the large-input benchmark |
| `SOFIE_BENCHMARK_LARGE_CUDA_ARCH` | `80` | CUDA architecture for the large-input benchmark |

---

## Build

Build the normal benchmark with:

```bash
cmake --build benchmark/build --target sofie_benchmark -j$(nproc)
```

The build first creates `sofie_benchmark_emitter`, which parses each selected ONNX model and generates:

```text
<Model>_GPU_ALPAKA.hxx
<Model>_GPU_ALPAKA.dat
<Model>_bench.hxx
<Model>_aot_meta.hxx
```

It then compiles the generated model code into `sofie_benchmark`.

---

## Run

Move to the benchmark build directory:

```bash
cd benchmark/build/benchmark
```

SOFIE only:

```bash
./sofie_benchmark
```

SOFIE + ONNX Runtime:

```bash
./sofie_benchmark --onnxruntime
```

SOFIE + TensorRT:

```bash
./sofie_benchmark --tensorrt
```

SOFIE + PyTorch AOTInductor:

```bash
./sofie_benchmark --aot
```

All available comparison backends:

```bash
./sofie_benchmark \
  --onnxruntime \
  --tensorrt \
  --aot
```

Each model/backend is run in a separate subprocess so that it starts with a fresh CUDA context.

A timestamped result directory is created under:

```text
benchmark/results/benchmark_<YYYYMMDD_HHMMSS>/
```

Depending on the enabled runtime comparisons it contains:

```text
benchmark_<timestamp>/
├── sofie/
│   └── benchmark.csv
├── ort/
│   └── benchmark.csv
├── tensorrt/
│   └── benchmark.csv
└── pytorch_aot/
    └── benchmark.csv
```

---

## GPU Occupancy (Currently supported in NVIDIA GPUs only)

Alongside latency/throughput, the default (non-profiling) benchmark table reports how
busy the GPU actually was **during the timed inference loop**, via a background thread
sampling NVML while the loop runs.

| Column | Meaning |
|---|---|
| `SM%(avg)` | Average SM (compute) activity duty-cycle across all samples taken during the timed inference loop |
| `SM%(pk)`  | Peak single-sample SM activity observed during the loop |
| `MemBw%(avg)` | Average memory-controller activity duty-cycle |

Low `SM%(avg)` with a low `infer(ms)` usually means the current inference approach is
**launch-overhead or transfer bound** rather than compute bound — the GPU sits mostly
idle between kernel launches. Consistently high `SM%` values mean the model is actually
saturating the device's compute; in that regime, further speedups need algorithmic or
kernel-level work, not just reducing launch count.

### How it's measured (NVIDIA/NVML only)

This is **NVIDIA-specific driver-side polling, not GPU-side instrumentation** — no
counters are read from inside the generated kernels.
`GpuOccupancySampler` (`src/GpuOccupancy.hxx`) does, per model:

1. `occSampler.start()` is called immediately before the timed inference loop begins
   (after warmup, right at `t0_infer`) and spawns one `std::thread`.
2. That thread polls NVML's `nvmlDeviceGetUtilizationRates()` every **2 ms** for as long
   as the loop is running. Each call returns the NVIDIA driver's own counters:
   - `u.gpu` — % of the last NVML sampling window during which **any** kernel was
     executing on an SM (compute activity)
   - `u.memory` — % of that window during which the memory controller was active
3. Every poll accumulates into running sums and a running max; `occSampler.stop()` (called
   right after `t1_infer`) joins the thread.
4. `SM%(avg)` / `MemBw%(avg)` = (sum of samples) / (sample count); `SM%(pk)` = the single
   highest `u.gpu` sample seen. These are computed in the benchmark process itself — no
   external tool (`nvidia-smi`, Nsight, …) is invoked.

Because the sampling and the CUDA calls being measured run concurrently on the host,
this measures **whole-loop duty-cycle over wall-clock time**, sampled at ~500 Hz — it is
not synchronized to individual kernel launches and cannot tell you per-kernel occupancy
(warps/registers active per SM). It answers "was the GPU busy while this ran", not "how
efficiently was each kernel using the SM".

> **Caveat:** for models whose `infer(ms)` is much shorter than NVML's own sampling
> window, increase `--iterations` so the background thread collects enough samples for a
> representative average. Columns print `N/A` when occupancy sampling isn't available at
> all — either `libnvidia-ml` wasn't found at configure time (see the CMake message when
> configuring), or (on a non-CUDA backend) NVML was never wired up in the first place —
> rather than a misleading `0.0`.

---

## Runtime Options

| Flag | Default | Description |
|---|---|---|
| `--warmup, -w <N>` | `10` | Warm-up iterations |
| `--iterations, -n <N>` | `100` | Timed iterations |
| `--weights-dir <path>` | `.` | Directory containing SOFIE `.dat` weight files |
| `--onnxruntime, --ort` | off | Run ONNX Runtime GPU comparison |
| `--tensorrt, --trt` | off | Run TensorRT comparison |
| `--pytorch-aot, --aot` | off | Run PyTorch AOTInductor comparison |
| `--profile` | off | Run Nsight Compute cross-backend profiling |
| `--ncu <path>` | — | Path to the Nsight Compute executable; required with `--profile` |
| `--sofie-only` | off | Run only SOFIE |
| `--ort-only` | off | Run only ONNX Runtime |
| `--trt-only` | off | Run only TensorRT |
| `--aot-only` | off | Run only PyTorch AOTInductor |
| `--help, -h` | — | Print command-line help |

`--single-model <name>` is an internal option used by the parent benchmark process to isolate each model/backend execution in its own subprocess.

The runtime `--profile` option is **not** the same as the CMake option `SOFIE_BENCHMARK_PROFILE`.

- `SOFIE_BENCHMARK_PROFILE=ON` enables SOFIE's internal per-operator profiler.
- `--profile --ncu <path>` launches the separate Nsight Compute cross-backend profiling workflow.

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

In an internal profiling build, each model prints two profiling blocks:

**GPU Profiling Results** — per-operator wall-clock time (microseconds) measured
with `std::chrono` and an `alpaka::wait(queue)` synchronisation point after every
kernel.  Results are sorted by average time descending, with ± stderr over all
timed iterations.  Warmup iterations are excluded (the session is reset before the
timed runs start).


**Memory Usage Breakdown** — sizes computed at code-generation time from tensor
shapes and types.  No runtime measurement is needed; the values are embedded
as constants in the generated session code.


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

Nsight Compute can profile **SOFIE, ONNX Runtime GPU, TensorRT, and PyTorch AOTInductor** using the same benchmark models.

This is separate from the internal SOFIE profiler described above.

For Nsight Compute profiling, configure with:

```text
SOFIE_BENCHMARK_PROFILE=OFF
```

The runtime `--profile` flag starts the Nsight Compute workflow. Each backend is run in a separate subprocess and one steady-state inference is captured.

### 1. Configure and build

For all comparison backends:

```bash
cmake -B benchmark/build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DSOFIE_BENCHMARK_TRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DSOFIE_BENCHMARK_AOT=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DSOFIE_BENCHMARK_PROFILE=OFF \
  -DSOFIE_BENCHMARK_CUDA_ARCH=<sm> \
  .
```

Replace `<sm>` with the CUDA architecture for the target GPU.

Build:

```bash
cmake --build benchmark/build --target sofie_benchmark -j$(nproc)
```

If Mamba should also be profiled through AOTInductor, add:

```bash
-DSOFIE_BENCHMARK_MAMBA_AOT=ON
```

and, if automatic selective-scan detection fails:

```bash
-DMAMBA_SELECTIVE_SCAN_LIBRARY=/path/to/selective_scan_cuda.so
```

All required AOT `.pt2` packages must already exist before starting profiling.

### 2. Generate TensorRT engines before profiling

TensorRT engine construction should not be performed inside Nsight Compute because engine generation itself executes GPU work that Nsight may instrument and replay.

Run TensorRT once normally:

```bash
cd benchmark/build/benchmark
./sofie_benchmark --tensorrt
```

This creates the serialized TensorRT engine cache.

Subsequent profiling runs load the cached plans instead of rebuilding them.

### 3. Run Nsight Compute

Nsight Compute may require administrator privileges to access GPU performance counters.

From `benchmark/build/benchmark`:

```bash
sudo ./sofie_benchmark \
  --profile \
  --onnxruntime \
  --tensorrt \
  --aot \
  --ncu /usr/local/cuda/bin/ncu
```

Use the actual path to `ncu` if it is installed elsewhere.

The benchmark uses:

```text
--profile-from-start off
```

so backend/model initialisation and the priming inference are excluded from the captured kernel statistics.

### 4. Profiling output

Each run creates:

```text
benchmark/results/benchmark_<YYYYMMDD_HHMMSS>/
```

A full four-backend run has the following layout:

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
├── tensorrt/
│   ├── benchmark.csv
│   ├── <model>.ncu-rep
│   └── <model>.csv
└── pytorch_aot/
    ├── benchmark.csv
    ├── <model>.ncu-rep
    └── <model>.csv
```

The `.ncu-rep` files contain the full Nsight Compute reports.

The per-model `.csv` files contain the exported metrics consumed by the profiling-summary script.

### 5. Generate the profiling summary

From the SOFIE repository root:

```bash
python3 benchmark/src/summarize_profile.py \
  benchmark/results/benchmark_<YYYYMMDD_HHMMSS>
```

The script generates:

```text
benchmark/results/benchmark_<timestamp>/profile_summary.md
benchmark/results/benchmark_<timestamp>/profile_summary.json
```

The summary includes:

- kernel-launch counts;
- unique kernel counts;
- total GPU kernel time;
- average and maximum kernel duration;
- register usage;
- achieved occupancy;
- shared-memory usage;
- DRAM throughput and bandwidth;
- L1 and L2 hit rates;
- spilling detection;
- Tensor Core activity;
- likely memory-bound and compute-bound launches;
- per-model summaries;
- slowest kernels;
- highest-register kernels;
- lowest-occupancy kernels.

The JSON file contains the same information in machine-readable form.

### 6. Important measurement note

Do not use latency, throughput, transfer-time, or peak-memory values from an Nsight Compute run as normal benchmark measurements.

Nsight Compute instruments and may replay kernels, so the wall-clock execution time is intentionally distorted.

Use a normal benchmark run for end-to-end performance:

```bash
./sofie_benchmark \
  --onnxruntime \
  --tensorrt \
  --aot
```

Use a separate Nsight run for kernel-level statistics:

```bash
sudo ./sofie_benchmark \
  --profile \
  --onnxruntime \
  --tensorrt \
  --aot \
  --ncu /usr/local/cuda/bin/ncu
```

The total GPU time reported in the Nsight summary is the sum of captured kernel durations and is not necessarily equal to end-to-end inference latency.

---

## Plotting Benchmark Results

Normal benchmark runs can be plotted with:

```bash
python3 benchmark/src/plot_results.py \
  benchmark/results/benchmark_<YYYYMMDD_HHMMSS>
```

The plotting script reads the backend `benchmark.csv` files and writes its output to:

```text
benchmark/results/benchmark_<timestamp>/plots/
```

It generates PNG and PDF figures together with:

```text
aggregated_results.csv
comparison_summary.csv
```

The plotting code recognises:

```text
sofie
ort
tensorrt
pytorch_aot
```

Multiple benchmark runs can be supplied to average their results and compute standard deviations:

```bash
python3 benchmark/src/plot_results.py \
  benchmark/results/benchmark_<run1> \
  benchmark/results/benchmark_<run2> \
  benchmark/results/benchmark_<run3>
```

---

## Re-running After Adding Models

Because model discovery happens during CMake configuration, re-run CMake after adding or removing ONNX models:

```bash
cmake -S . -B benchmark/build
cmake --build benchmark/build --target sofie_benchmark -j$(nproc)
```

---