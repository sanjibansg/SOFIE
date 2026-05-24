# SOFIE Alpaka Benchmark Toolkit

Measures **inference latency and throughput** for ONNX models compiled by SOFIE and
executed via Alpaka (CUDA backend).  Optionally runs the same models through
**ONNX Runtime GPU** for a side-by-side comparison.

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
# SOFIE inference only (default)
cmake -B build -DSOFIE_BENCHMARK=ON /path/to/SOFIE

# With ONNX Runtime GPU comparison
cmake -B build \
  -DSOFIE_BENCHMARK=ON \
  -DSOFIE_BENCHMARK_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  /path/to/SOFIE
```

| CMake flag | Default | Description |
|---|---|---|
| `-DSOFIE_BENCHMARK=ON` | — | Enable the benchmark suite |
| `-DSOFIE_BENCHMARK_ORT=ON` | `OFF` | Also benchmark ONNX Runtime GPU |
| `-DONNXRUNTIME_ROOT=<path>` | — | Hint for finding ORT headers/library |

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
   - `<Model>_GPU_ALPAKA.hxx` — SOFIE CUDA/Alpaka inference code
   - `<Model>_GPU_ALPAKA.dat` — serialized weights
   - `<Model>_bench.hxx`      — timing wrapper `Benchmark_<Model>()`
2. Builds **`sofie_benchmark`** — compiles all generated code as `.cu` and links the
   timing loop.

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

## Re-running after adding models

```bash
cmake build                                               # re-configure (re-globs)
cmake --build build --target sofie_benchmark -j$(nproc)  # re-build
```
