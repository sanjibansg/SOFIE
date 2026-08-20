cmake_minimum_required(VERSION 3.18)
include(FetchContent)

################################################################################
# SOFIE Alpaka Benchmark Toolkit
#
# Usage:
#   cmake -Bbuild -DSOFIE_BENCHMARK=ON .
#   cmake --build build --target sofie_benchmark
#   cd build/benchmark && ./sofie_benchmark [options]
#
# To also benchmark with ONNX Runtime GPU:
#   cmake -Bbuild -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_ORT=ON \
#         [-DONNXRUNTIME_ROOT=/usr/local/onnxruntime] .
#   ./sofie_benchmark --onnxruntime
#
# Place .onnx models in benchmark/models/ and re-run cmake to register them.
################################################################################

option(SOFIE_BENCHMARK_ORT
    "Also benchmark ONNX Runtime GPU alongside SOFIE (requires ORT ≥ 1.18)"
    OFF)

option(SOFIE_BENCHMARK_PROFILE
    "Enable per-operator GPU profiling (per-op timing + memory breakdown). \
Mutually exclusive with throughput benchmarking — rebuild without this flag \
to measure throughput."
    OFF)

option(SOFIE_BENCHMARK_LOWRANK
    "Emit all benchmark models with low rank factorization enabled \
(Options::kLowRankFactorize), so the same benchmark suite can be compared \
dense vs low-rank across two builds. Only affects the GPU codegen emitted \
for each model, not the sofie_benchmark runner itself."
    OFF)

set(SOFIE_BENCHMARK_LOWRANK_RATIO "0.5" CACHE STRING
    "Rank ratio used when SOFIE_BENCHMARK_LOWRANK is ON: rank = ratio * min(rows, cols) \
for each eligible Gemm weight matrix.")

################################################################################
# Backend / architecture selection
#
# SOFIE_BENCHMARK_BACKEND applies to both throughput benchmarking and profiling.
# Currently only CUDA is supported; setting any other value is a hard error.
################################################################################

set(SOFIE_BENCHMARK_BACKEND "CUDA" CACHE STRING
    "Target accelerator backend for SOFIE benchmark and profiling (currently only CUDA)")
set_property(CACHE SOFIE_BENCHMARK_BACKEND PROPERTY STRINGS CUDA)

string(TOUPPER "${SOFIE_BENCHMARK_BACKEND}" _bench_backend)

if(NOT _bench_backend STREQUAL "CUDA")
    message(FATAL_ERROR
        "SOFIE Benchmark: SOFIE_BENCHMARK_BACKEND='${SOFIE_BENCHMARK_BACKEND}' is not "
        "supported.  Only 'CUDA' is currently implemented.  "
        "AMD GPU (HIP/ROCm) support is planned for a future release.")
endif()

# Compile-time defines propagated to every benchmark translation unit.
# These drive BenchmarkBackend.hxx type aliases and SOFIE_BENCH_DEVICE_SYNC().
set(_SOFIE_BENCH_ALPAKA_DEFINE ALPAKA_ACC_GPU_CUDA_ENABLED)
set(_SOFIE_BENCH_BACKEND_DEFINE SOFIE_BACKEND_CUDA)

if(SOFIE_BENCHMARK_PROFILE)
    message(STATUS "SOFIE Benchmark: profiling ENABLED  "
                   "(backend = ${SOFIE_BENCHMARK_BACKEND}, throughput benchmarking disabled)")
else()
    message(STATUS "SOFIE Benchmark: backend = ${SOFIE_BENCHMARK_BACKEND}")
endif()

if(SOFIE_BENCHMARK_LOWRANK)
    message(STATUS "SOFIE Benchmark: low rank factorization ENABLED for all emitted models "
                   "(ratio = ${SOFIE_BENCHMARK_LOWRANK_RATIO}). Rebuild with -DSOFIE_BENCHMARK_LOWRANK=OFF "
                   "and re-run to compare against the dense baseline.")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

################################################################################
# Discover models
################################################################################

file(GLOB BENCHMARK_ONNX_MODELS
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.onnx")

list(FILTER BENCHMARK_ONNX_MODELS EXCLUDE REGEX
    "/(gnn_h32_k2|gnn_h64_k4|punet_h32_k2_heads4_layers2|punet_h64_k4_heads4_layers2|transformer_d32_h2_L6_ff32)\\.onnx$")

# Exclude models that exceed GPU memory (cudaErrorMemoryAllocation) or trigger
# a cuBLAS EXECUTION_FAILED error (status 13) on architectures that are not very powerful. 
# They stay on disk for reference and are
# compiled into sofie_benchmark_large (cluster target) instead.
list(FILTER BENCHMARK_ONNX_MODELS EXCLUDE REGEX
    "/(punet_h32_k2_heads4_layers2_n10000_e50000\
|punet_h32_k2_heads4_layers2_n1000_e5000\
|punet_h32_k2_heads4_layers2_n3000_e15000\
|punet_h32_k2_heads4_layers2_n30000_e150000\
|punet_h32_k2_heads4_layers2_n100000_e500000\
|punet_h64_k4_heads4_layers2_n10000_e50000\
|punet_h64_k4_heads4_layers2_n100_e500\
|punet_h64_k4_heads4_layers2_n300_e1500\
|punet_h64_k4_heads4_layers2_n3000_e15000\
|punet_h64_k4_heads4_layers2_n30000_e150000\
|punet_h64_k4_heads4_layers2_n100000_e500000\
|gnn_h32_k2_n30000_e150000\
|gnn_h32_k2_n100000_e500000\
|gnn_h64_k4_n30000_e150000\
|gnn_h64_k4_n100000_e500000\
|transformer_L1000_B100\
|transformer_L100_B100\
|transformer_L8000_B1\
|transformer_d32_h2_L6_ff32_n60_s60)\\.onnx$")

if(NOT BENCHMARK_ONNX_MODELS)
    message(STATUS
        "SOFIE Benchmark: No .onnx models found in benchmark/models/. "
        "Add ONNX models there and re-run cmake to enable benchmarking.")
    return()
endif()

list(LENGTH BENCHMARK_ONNX_MODELS N_MODELS)
message(STATUS "SOFIE Benchmark: Found ${N_MODELS} model(s) in benchmark/models/")


FetchContent_Declare(
    sofieBLAS
    GIT_REPOSITORY https://github.com/ML4EP/sofieBLAS
    GIT_TAG        dev
)
FetchContent_MakeAvailable(sofieBLAS)

FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka
    GIT_TAG        2fa91a34ed11b2076e474c5507d920e85cf9b79d
)
FetchContent_MakeAvailable(alpaka)

################################################################################
# Hardware toolkit setup — CUDA (the only supported backend for now)
################################################################################

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "SOFIE Benchmark: CUDA toolkit ${CUDAToolkit_VERSION}")

################################################################################
# NVML — background GPU SM/memory occupancy sampling during the timed
# inference loop (see src/GpuOccupancy.hxx). Ships as a stub library
# alongside libcuda.so in the CUDA toolkit; the real libnvidia-ml.so.1 is
# resolved at runtime from the NVIDIA driver install, same pattern as the
# CUDA driver stub.
################################################################################

find_library(SOFIE_NVML_LIBRARY
    NAMES nvidia-ml
    PATHS "${CUDAToolkit_LIBRARY_DIR}/stubs" "${CUDAToolkit_LIBRARY_DIR}"
)

if(SOFIE_NVML_LIBRARY)
    message(STATUS "SOFIE Benchmark: NVML found — ${SOFIE_NVML_LIBRARY} (GPU occupancy sampling enabled)")
else()
    message(WARNING
        "SOFIE Benchmark: NVML (libnvidia-ml) not found — GPU occupancy "
        "sampling will be disabled; SM%/MemBw% columns will print N/A.")
endif()


set(SOFIE_ORT_FOUND FALSE)

if(SOFIE_BENCHMARK_ORT)
    # Prefer manual detection — the installed ORT CMake config may reference
    # a wrong lib path (e.g. lib64 vs lib) and raise a hard error even with QUIET.
    # If ONNXRUNTIME_ROOT is provided, go straight to the manual path.
    # Otherwise attempt the CMake config with NO_DEFAULT_PATH so it only looks
    # where we tell it, and fall through to manual on failure.

    set(_ort_search_roots "")
    if(DEFINED ONNXRUNTIME_ROOT)
        list(APPEND _ort_search_roots "${ONNXRUNTIME_ROOT}")
    endif()
    list(APPEND _ort_search_roots
        /usr/local/onnxruntime /usr/local /usr /opt)

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${_ort_search_roots}
        PATH_SUFFIXES include include/onnxruntime
        NO_DEFAULT_PATH)

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${_ort_search_roots}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(SOFIE_ORT_FOUND TRUE)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION             "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
        set(SOFIE_ORT_TARGET onnxruntime::onnxruntime)
        message(STATUS "SOFIE Benchmark: ONNX Runtime found — ${ONNXRUNTIME_LIBRARY}")
        message(STATUS "SOFIE Benchmark: ORT headers  — ${ONNXRUNTIME_INCLUDE_DIR}")
    else()
        message(WARNING
            "SOFIE Benchmark: SOFIE_BENCHMARK_ORT=ON but ONNX Runtime "
            "not found.  Set -DONNXRUNTIME_ROOT=<path> or install ORT.  "
            "ORT benchmarking will be disabled.")
    endif()
endif()

if(SOFIE_BENCHMARK_ORT AND NOT SOFIE_ORT_FOUND)
    message(STATUS "SOFIE Benchmark: ORT benchmarking disabled (library not found)")
endif()

set(_EMIT_BLOCK
"try {\n\
    EmitBenchmarkModel(\"@1@\", \"@2@\", outDir);\n\
} catch (const std::exception &e) {\n\
    std::cerr << \"[ERROR] @2@: \" << e.what() << \"\\n\";\n\
    ++failures;\n\
} catch (...) {\n\
    std::cerr << \"[ERROR] @2@: unknown exception\\n\";\n\
    ++failures;\n\
}\n\
")

set(BENCHMARK_EMIT_CAPTURES        "")
set(BENCHMARK_BENCH_HEADERS        "")
set(BENCHMARK_FWD_DECLS            "")
set(BENCHMARK_SINGLE_MODEL_CASES   "")
set(BENCHMARK_SPAWN_CALLS          "")
set(GENERATED_HEADERS              "")
set(BENCHMARK_MODEL_CU_SRCS        "")

foreach(ONNX_FILE ${BENCHMARK_ONNX_MODELS})
    get_filename_component(MODEL_NAME "${ONNX_FILE}" NAME_WE)

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" MODEL_CPPNAME "${MODEL_NAME}")

    set(GEN_HXX   "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_GPU_ALPAKA.hxx")
    set(GEN_BENCH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.hxx")
    list(APPEND GENERATED_HEADERS "${GEN_HXX}" "${GEN_BENCH}")

    string(REPLACE "@1@" "${ONNX_FILE}"  _emit_cap "${_EMIT_BLOCK}")
    string(REPLACE "@2@" "${MODEL_NAME}" _emit_cap "${_emit_cap}")
    string(APPEND BENCHMARK_EMIT_CAPTURES "${_emit_cap}")

    string(APPEND BENCHMARK_BENCH_HEADERS
        "#include \"${MODEL_NAME}_bench.hxx\"\n")

    # Forward declaration for the main TU (function defined in per-model .cu)
    string(APPEND BENCHMARK_FWD_DECLS
        "void Benchmark_${MODEL_CPPNAME}(int warmup, int iterations, const std::string& weightsDir);\n")

    # Single-model dispatch: one if-branch per model (used in --single-model mode)
    string(APPEND BENCHMARK_SINGLE_MODEL_CASES
        "    if (model == \"${MODEL_CPPNAME}\") {\n"
        "        Benchmark_${MODEL_CPPNAME}(warmup, iterations, weightsDir);\n"
        "#ifdef SOFIE_BENCHMARK_ORT\n"
        "        if (run_ort) BenchmarkORT_GPU(\"${ONNX_FILE}\", \"${MODEL_NAME}\", warmup, iterations);\n"
        "#endif\n"
        "        return 0;\n"
        "    }\n"
    )

    # Subprocess spawn: run this binary as a child with --single-model <name>.
    # Each child gets a fresh CUDA context so GPU memory from the previous
    # model is completely freed before the next one allocates.
    string(APPEND BENCHMARK_SPAWN_CALLS
        "    {\n"
        "        std::string cmd = std::string(argv[0])\n"
        "            + \" --single-model ${MODEL_CPPNAME}\"\n"
        "            + commonArgs;\n"
        "        int rc = std::system(cmd.c_str());\n"
        "        if (rc != 0) {\n"
        "            std::fprintf(stderr, \"[ERROR] ${MODEL_NAME}: subprocess exited %d\\n\", rc);\n"
        "            ++totalFailed;\n"
        "        }\n"
        "    }\n"
    )

    # Per-model compilation unit: include one bench.hxx → one .cu file
    set(_model_cu "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.cu")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ModelBench.cu.in"
        "${_model_cu}"
        @ONLY
    )
    list(APPEND BENCHMARK_MODEL_CU_SRCS "${_model_cu}")
endforeach()

################################################################################
# Configure emitter and runner sources
################################################################################

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkEmitter.cxx.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_all.cxx"
    @ONLY
)

# Main runner: only main() + forward declarations (no model headers).
# Each model is compiled in its own _bench.cu TU (see BENCHMARK_MODEL_CU_SRCS).
set(RUNNER_SRC "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_main.cpp")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkRunner.cxx.in"
    "${RUNNER_SRC}"
    @ONLY
)

################################################################################
# Emitter executable (plain C++, generates SOFIE headers at build time)
################################################################################

add_executable(sofie_benchmark_emitter
    "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_all.cxx"
)

target_include_directories(sofie_benchmark_emitter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../core/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc"
)

target_link_libraries(sofie_benchmark_emitter PRIVATE
    SOFIE_core
    SOFIE_parsers
    protobuf::libprotobuf
)

target_compile_options(sofie_benchmark_emitter PRIVATE
    -Wno-unused-parameter
    -Wno-array-bounds
)

target_compile_definitions(sofie_benchmark_emitter PRIVATE
    $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
    $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK>
    $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK_RATIO=${SOFIE_BENCHMARK_LOWRANK_RATIO}>
)


add_custom_command(
    OUTPUT  ${GENERATED_HEADERS}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "$<TARGET_FILE:sofie_benchmark_emitter>" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS sofie_benchmark_emitter ${BENCHMARK_ONNX_MODELS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "SOFIE Benchmark: generating headers for ${N_MODELS} model(s)..."
    VERBATIM
)

add_custom_target(sofie_benchmark_headers
    DEPENDS ${GENERATED_HEADERS}
)

set_source_files_properties(${BENCHMARK_MODEL_CU_SRCS} PROPERTIES LANGUAGE CUDA)

set_source_files_properties("${RUNNER_SRC}" PROPERTIES LANGUAGE CXX)

add_executable(sofie_benchmark "${RUNNER_SRC}" ${BENCHMARK_MODEL_CU_SRCS})

add_dependencies(sofie_benchmark sofie_benchmark_headers)

target_include_directories(sofie_benchmark PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${alpaka_SOURCE_DIR}/include"
    "${sofieblas_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)


if(NOT DEFINED SOFIE_BENCHMARK_CUDA_ARCH OR SOFIE_BENCHMARK_CUDA_ARCH STREQUAL "")
    if(CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(SOFIE_BENCHMARK_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(SOFIE_BENCHMARK_CUDA_ARCH "75")   # RTX 2070 SUPER default
    endif()
endif()
message(STATUS "SOFIE Benchmark: CUDA architectures = ${SOFIE_BENCHMARK_CUDA_ARCH}")

set_target_properties(sofie_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES          "${SOFIE_BENCHMARK_CUDA_ARCH}"
    CUDA_STANDARD               20
    CUDA_STANDARD_REQUIRED      ON
    RUNTIME_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_definitions(sofie_benchmark PRIVATE
    ${_SOFIE_BENCH_ALPAKA_DEFINE}
    ${_SOFIE_BENCH_BACKEND_DEFINE}
    ALPAKA_HAS_STD_ATOMIC_REF
    $<$<BOOL:${SOFIE_ORT_FOUND}>:SOFIE_BENCHMARK_ORT>
    $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
    $<$<BOOL:${SOFIE_NVML_LIBRARY}>:SOFIE_BENCH_HAVE_NVML>
)

target_compile_options(sofie_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        -O1
        -Wno-deprecated-gpu-targets
        -diag-suppress 177
        --maxrregcount=64
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O2
        -fPIC
    >
)

find_package(Threads REQUIRED)

target_link_libraries(sofie_benchmark PRIVATE
    SOFIE_core
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    Threads::Threads
    $<$<BOOL:${SOFIE_ORT_FOUND}>:${SOFIE_ORT_TARGET}>
    $<$<BOOL:${SOFIE_NVML_LIBRARY}>:${SOFIE_NVML_LIBRARY}>
)

if(SOFIE_ORT_FOUND)
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend + ORT-GPU)")
else()
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend; "
                   "re-configure with -DSOFIE_BENCHMARK_ORT=ON for ORT comparison)")
endif()

if(testing)
    add_test(
        NAME    SofieBenchmark
        COMMAND sofie_benchmark --warmup 5 --iterations 20
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

################################################################################
# Large-input benchmark (cluster GPUs: A100 / H100 / MI300X, ≥40 GB VRAM)
#
# Enables:
#  • All models excluded from sofie_benchmark due to OOM on ≤8 GB GPUs
#  • New very-large GNN / PUNet variants: n=30 000 / n=100 000 nodes
#  • transformer_d32_h2_L6_ff32_n60_s60  (max sequence length for this arch)
#
# Usage:
#   cmake -Bbuild . -DSOFIE_BENCHMARK=ON -DSOFIE_BENCHMARK_LARGE=ON \
#         [-DSOFIE_BENCHMARK_LARGE_CUDA_ARCH=80]   # 80=A100, 90=H100
#   cmake --build build --target sofie_benchmark_large
################################################################################

option(SOFIE_BENCHMARK_LARGE
    "Build sofie_benchmark_large for cluster GPUs (A100/H100, ≥40 GB VRAM)"
    OFF)

if(SOFIE_BENCHMARK_LARGE)

    set(SOFIE_BENCHMARK_LARGE_CUDA_ARCH "80" CACHE STRING
        "CUDA SM architecture(s) for sofie_benchmark_large (e.g. 80=A100, 90=H100)")

    set(LARGE_MODEL_NAMES
        # ── PUNet h32/k2  (OOM + new) ──────────────────────────────────────
        "punet_h32_k2_heads4_layers2_n1000_e5000"
        "punet_h32_k2_heads4_layers2_n3000_e15000"
        "punet_h32_k2_heads4_layers2_n10000_e50000"
        "punet_h32_k2_heads4_layers2_n30000_e150000"
        "punet_h32_k2_heads4_layers2_n100000_e500000"
        # ── PUNet h64/k4  (OOM + new) ──────────────────────────────────────
        "punet_h64_k4_heads4_layers2_n100_e500"
        "punet_h64_k4_heads4_layers2_n300_e1500"
        "punet_h64_k4_heads4_layers2_n1000_e5000"
        "punet_h64_k4_heads4_layers2_n3000_e15000"
        "punet_h64_k4_heads4_layers2_n10000_e50000"
        "punet_h64_k4_heads4_layers2_n30000_e150000"
        "punet_h64_k4_heads4_layers2_n100000_e500000"
        # ── GNN h32/k2  (new large) ────────────────────────────────────────
        "gnn_h32_k2_n30000_e150000"
        "gnn_h32_k2_n100000_e500000"
        # ── GNN h64/k4  (new large) ────────────────────────────────────────
        "gnn_h64_k4_n30000_e150000"
        "gnn_h64_k4_n100000_e500000"
        # ── Transformers (OOM on 8 GB) ──────────────────────────────────────
        "transformer_L100_B100"
        "transformer_L1000_B100"
        "transformer_L8000_B1"
        # ── Transformer d32 (max sequence length for this architecture) ─────
        "transformer_d32_h2_L6_ff32_n60_s60"
    )

    set(_large_regex "")
    foreach(_m ${LARGE_MODEL_NAMES})
        if(_large_regex)
            set(_large_regex "${_large_regex}|${_m}")
        else()
            set(_large_regex "${_m}")
        endif()
    endforeach()
    set(_large_regex "(${_large_regex})\\.onnx$")

    file(GLOB _ALL_LARGE_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/models/*.onnx")
    list(FILTER _ALL_LARGE_ONNX INCLUDE REGEX "${_large_regex}")

    if(NOT _ALL_LARGE_ONNX)
        message(WARNING
            "SOFIE_BENCHMARK_LARGE=ON but none of the expected large-input .onnx "
            "files were found in benchmark/models/.  Run specialize_models.py "
            "(with the extended GNN_VARIANTS list) first, then re-run cmake.")
    else()
        list(LENGTH _ALL_LARGE_ONNX N_LARGE)
        message(STATUS
            "SOFIE Benchmark Large: ${N_LARGE} large-input model(s), "
            "CUDA arch = ${SOFIE_BENCHMARK_LARGE_CUDA_ARCH}")

        set(LARGE_EMIT_CAPTURES      "")
        set(LARGE_FWD_DECLS          "")
        set(LARGE_SINGLE_MODEL_CASES "")
        set(LARGE_SPAWN_CALLS        "")
        set(LARGE_GENERATED_HEADERS  "")
        set(LARGE_MODEL_CU_SRCS      "")

        foreach(ONNX_FILE ${_ALL_LARGE_ONNX})
            get_filename_component(MODEL_NAME "${ONNX_FILE}" NAME_WE)
            string(REGEX REPLACE "[^A-Za-z0-9]" "_" MODEL_CPPNAME "${MODEL_NAME}")

            set(GEN_HXX   "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_GPU_ALPAKA.hxx")
            set(GEN_BENCH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.hxx")
            list(APPEND LARGE_GENERATED_HEADERS "${GEN_HXX}" "${GEN_BENCH}")

            string(REPLACE "@1@" "${ONNX_FILE}"  _ec "${_EMIT_BLOCK}")
            string(REPLACE "@2@" "${MODEL_NAME}" _ec "${_ec}")
            string(APPEND LARGE_EMIT_CAPTURES "${_ec}")

            string(APPEND LARGE_FWD_DECLS
                "void Benchmark_${MODEL_CPPNAME}(int warmup, int iterations, const std::string& weightsDir);\n")

            string(APPEND LARGE_SINGLE_MODEL_CASES
                "    if (model == \"${MODEL_CPPNAME}\") {\n"
                "        Benchmark_${MODEL_CPPNAME}(warmup, iterations, weightsDir);\n"
                "#ifdef SOFIE_BENCHMARK_ORT\n"
                "        if (run_ort) BenchmarkORT_GPU(\"${ONNX_FILE}\", \"${MODEL_NAME}\", warmup, iterations);\n"
                "#endif\n"
                "        return 0;\n"
                "    }\n"
            )

            string(APPEND LARGE_SPAWN_CALLS
                "    {\n"
                "        std::string cmd = std::string(argv[0])\n"
                "            + \" --single-model ${MODEL_CPPNAME}\"\n"
                "            + commonArgs;\n"
                "        int rc = std::system(cmd.c_str());\n"
                "        if (rc != 0) {\n"
                "            std::fprintf(stderr, \"[ERROR] ${MODEL_NAME}: subprocess exited %d\\n\", rc);\n"
                "            ++totalFailed;\n"
                "        }\n"
                "    }\n"
            )

            set(_model_cu "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.cu")
            configure_file(
                "${CMAKE_CURRENT_SOURCE_DIR}/src/ModelBench.cu.in"
                "${_model_cu}"
                @ONLY
            )
            list(APPEND LARGE_MODEL_CU_SRCS "${_model_cu}")
        endforeach()

        set(BENCHMARK_EMIT_CAPTURES      "${LARGE_EMIT_CAPTURES}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkEmitter.cxx.in"
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_large.cxx"
            @ONLY
        )

        set(BENCHMARK_FWD_DECLS          "${LARGE_FWD_DECLS}")
        set(BENCHMARK_SINGLE_MODEL_CASES "${LARGE_SINGLE_MODEL_CASES}")
        set(BENCHMARK_SPAWN_CALLS        "${LARGE_SPAWN_CALLS}")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkRunner.cxx.in"
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            @ONLY
        )

        add_executable(sofie_benchmark_large_emitter
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_large.cxx"
        )
        target_include_directories(sofie_benchmark_large_emitter PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../core/inc"
            "${CMAKE_CURRENT_SOURCE_DIR}/../parsers/inc"
        )
        target_link_libraries(sofie_benchmark_large_emitter PRIVATE
            SOFIE_core
            SOFIE_parsers
            protobuf::libprotobuf
        )
        target_compile_options(sofie_benchmark_large_emitter PRIVATE
            -Wno-unused-parameter
            -Wno-array-bounds
        )
        target_compile_definitions(sofie_benchmark_large_emitter PRIVATE
            $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
            $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK>
            $<$<BOOL:${SOFIE_BENCHMARK_LOWRANK}>:SOFIE_BENCHMARK_LOWRANK_RATIO=${SOFIE_BENCHMARK_LOWRANK_RATIO}>
        )

        add_custom_command(
            OUTPUT  ${LARGE_GENERATED_HEADERS}
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
            COMMAND "$<TARGET_FILE:sofie_benchmark_large_emitter>" "${CMAKE_CURRENT_BINARY_DIR}"
            DEPENDS sofie_benchmark_large_emitter ${_ALL_LARGE_ONNX}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "SOFIE Benchmark Large: generating headers for ${N_LARGE} large model(s)..."
            VERBATIM
        )

        add_custom_target(sofie_benchmark_large_headers
            DEPENDS ${LARGE_GENERATED_HEADERS}
        )

        set_source_files_properties(${LARGE_MODEL_CU_SRCS} PROPERTIES LANGUAGE CUDA)
        set_source_files_properties(
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            PROPERTIES LANGUAGE CXX)

        add_executable(sofie_benchmark_large
            "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_large.cpp"
            ${LARGE_MODEL_CU_SRCS}
        )
        add_dependencies(sofie_benchmark_large sofie_benchmark_large_headers)

        target_include_directories(sofie_benchmark_large PRIVATE
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${alpaka_SOURCE_DIR}/include"
            "${sofieblas_SOURCE_DIR}/include"
            "${CUDAToolkit_INCLUDE_DIRS}"
        )

        set_target_properties(sofie_benchmark_large PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES          "${SOFIE_BENCHMARK_LARGE_CUDA_ARCH}"
            CUDA_STANDARD               20
            CUDA_STANDARD_REQUIRED      ON
            RUNTIME_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_BINARY_DIR}"
        )

        target_compile_definitions(sofie_benchmark_large PRIVATE
            ${_SOFIE_BENCH_ALPAKA_DEFINE}
            ${_SOFIE_BENCH_BACKEND_DEFINE}
            ALPAKA_HAS_STD_ATOMIC_REF
            $<$<BOOL:${SOFIE_BENCHMARK_PROFILE}>:SOFIE_BENCHMARK_PROFILE>
            $<$<BOOL:${SOFIE_NVML_LIBRARY}>:SOFIE_BENCH_HAVE_NVML>
        )

        target_compile_options(sofie_benchmark_large PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --extended-lambda
                --expt-relaxed-constexpr
                --use_fast_math
                -O2
                -Wno-deprecated-gpu-targets
                -diag-suppress 177
                --maxrregcount=64
            >
            $<$<COMPILE_LANGUAGE:CXX>:
                -O2
                -fPIC
            >
        )

        target_link_libraries(sofie_benchmark_large PRIVATE
            SOFIE_core
            CUDA::cudart_static
            CUDA::cublas_static
            CUDA::cublasLt_static
            CUDA::culibos
            Threads::Threads
            $<$<BOOL:${SOFIE_NVML_LIBRARY}>:${SOFIE_NVML_LIBRARY}>
        )

        message(STATUS
            "SOFIE Benchmark Large: target 'sofie_benchmark_large' configured "
            "(${N_LARGE} large model(s), sm${SOFIE_BENCHMARK_LARGE_CUDA_ARCH})")
    endif() # _ALL_LARGE_ONNX
endif() # SOFIE_BENCHMARK_LARGE
