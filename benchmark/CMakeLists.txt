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

if(NOT BENCHMARK_ONNX_MODELS)
    message(STATUS
        "SOFIE Benchmark: No .onnx models found in benchmark/models/. "
        "Add ONNX models there and re-run cmake to enable benchmarking.")
    return()
endif()

list(LENGTH BENCHMARK_ONNX_MODELS N_MODELS)
message(STATUS "SOFIE Benchmark: Found ${N_MODELS} model(s) in benchmark/models/")

################################################################################
# Fetch Alpaka and sofieBLAS (same pinned revisions as the test suite)
################################################################################

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
# CUDA (same as the test suite)
################################################################################

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "SOFIE Benchmark: CUDA backend (${CUDAToolkit_VERSION})")

################################################################################
# Optional: ONNX Runtime GPU backend
################################################################################

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

    # Manual header + library search (reliable, no broken cmake-config risk)
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

################################################################################
# Build per-model strings for configure_file
################################################################################

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

set(_RUN_BLOCK
"    Benchmark_@3@(warmup, iterations, weightsDir);\n\
")

# ORT call: passes the full ONNX path directly (no SOFIE weights needed)
set(_ORT_RUN_BLOCK
"#ifdef SOFIE_BENCHMARK_ORT\n\
    if (run_ort) BenchmarkORT_GPU(\"@1@\", \"@2@\", warmup, iterations);\n\
#endif\n\
")

set(BENCHMARK_EMIT_CAPTURES "")
set(BENCHMARK_BENCH_HEADERS "")
set(BENCHMARK_RUN_CALLS     "")
set(GENERATED_HEADERS       "")

foreach(ONNX_FILE ${BENCHMARK_ONNX_MODELS})
    get_filename_component(MODEL_NAME "${ONNX_FILE}" NAME_WE)

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" MODEL_CPPNAME "${MODEL_NAME}")

    set(GEN_HXX   "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_GPU_ALPAKA.hxx")
    set(GEN_BENCH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_NAME}_bench.hxx")
    list(APPEND GENERATED_HEADERS "${GEN_HXX}" "${GEN_BENCH}")

    string(REPLACE "@1@" "${ONNX_FILE}"     _emit_cap "${_EMIT_BLOCK}")
    string(REPLACE "@2@" "${MODEL_NAME}"    _emit_cap "${_emit_cap}")
    string(APPEND BENCHMARK_EMIT_CAPTURES "${_emit_cap}")

    # SOFIE Alpaka call
    string(REPLACE "@3@" "${MODEL_CPPNAME}" _run_cap  "${_RUN_BLOCK}")
    string(APPEND BENCHMARK_RUN_CALLS "${_run_cap}")

    # ORT call (guarded by #ifdef at compile time + run_ort flag at runtime)
    string(REPLACE "@1@" "${ONNX_FILE}"  _ort_cap "${_ORT_RUN_BLOCK}")
    string(REPLACE "@2@" "${MODEL_NAME}" _ort_cap "${_ort_cap}")
    string(APPEND BENCHMARK_RUN_CALLS "${_ort_cap}")

    string(APPEND BENCHMARK_BENCH_HEADERS
        "#include \"${MODEL_NAME}_bench.hxx\"\n")
endforeach()

################################################################################
# Configure emitter and runner sources
################################################################################

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/BenchmarkEmitter.cxx.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkEmitter_all.cxx"
    @ONLY
)

set(RUNNER_SRC "${CMAKE_CURRENT_BINARY_DIR}/BenchmarkRunner_all.cu")
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

################################################################################
# Custom command: run emitter → generate inference + benchmark headers
################################################################################

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

################################################################################
# Benchmark runner (compiled as .cu, same as the test suite)
################################################################################

set_source_files_properties("${RUNNER_SRC}" PROPERTIES LANGUAGE CUDA)

add_executable(sofie_benchmark "${RUNNER_SRC}")

add_dependencies(sofie_benchmark sofie_benchmark_headers)

target_include_directories(sofie_benchmark PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"                 # generated headers live here
    "${CMAKE_CURRENT_SOURCE_DIR}/src"             # ONNXRuntimeBenchmark.hxx
    "${alpaka_SOURCE_DIR}/include"
    "${sofieblas_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

set_target_properties(sofie_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES          "70;75;80;86;89;90"
    CUDA_STANDARD               20
    CUDA_STANDARD_REQUIRED      ON
    RUNTIME_OUTPUT_DIRECTORY    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_definitions(sofie_benchmark PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
    ALPAKA_HAS_STD_ATOMIC_REF
    $<$<BOOL:${SOFIE_ORT_FOUND}>:SOFIE_BENCHMARK_ORT>
)

target_compile_options(sofie_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        -O2
        -Wno-deprecated-gpu-targets
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O2
        -fPIC
    >
)

target_link_libraries(sofie_benchmark PRIVATE
    SOFIE_core
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    $<$<BOOL:${SOFIE_ORT_FOUND}>:${SOFIE_ORT_TARGET}>
)

if(SOFIE_ORT_FOUND)
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend + ORT-GPU)")
else()
    message(STATUS "SOFIE Benchmark: target 'sofie_benchmark' configured "
                   "(${N_MODELS} model(s), CUDA backend; "
                   "re-configure with -DSOFIE_BENCHMARK_ORT=ON for ORT comparison)")
endif()

# Convenience CTest entry
if(testing)
    add_test(
        NAME    SofieBenchmark
        COMMAND sofie_benchmark --warmup 5 --iterations 20
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
