cmake_minimum_required(VERSION 3.14)
include(FetchContent)

############################################################################
# Basic setup
############################################################################
include_directories(${CMAKE_SOURCE_DIR}/core/inc)
include_directories(${CMAKE_SOURCE_DIR}/parsers/inc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT ONNX_MODELS_DIR)
  set(ONNX_MODELS_DIR input_models)
endif()

option(ENABLE_ALPAKA_TESTS "Enable Alpaka-based SOFIE tests" OFF)

set(ALPAKA_BACKEND "cuda"
    CACHE STRING "Alpaka backend to test (cuda, cpu, hip, sycl)")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS cuda cpu hip sycl)

############################################################################
# Generate emitter sources
############################################################################
set(CAPTURE_STR
"try {\n\
    EmitModel(\"@1\", \"@2\");\n\
} catch (const std::exception& e) {\n\
    std::string msg = e.what();\n\
    if (msg.find(\"multiple output tensors are not supported\") != std::string::npos) {\n\
        std::cerr << \"[SKIP] Multiple outputs are not supported for @1\" << std::endl;\n\
    } else if (msg.find(\"is of a data type which is not yet supported\") != std::string::npos) {\n\
        std::cerr << \"[SKIP] Operator with unsupported data type in @1: \" << msg << std::endl;\n\
    } else {\n\
        std::cerr << \"[ERROR] Failed processing @1: \" << msg << std::endl;\n\
        failures++;\n\
    }\n\
} catch (...) {\n\
    std::cerr << \"[ERROR] Unknown failure processing @1\" << std::endl;\n\
    failures++;\n\
}\n\
")

file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")

set(ALL_CAPTURES "")
foreach(onnx_file ${ONNX_FILES})
  get_filename_component(fname ${onnx_file} NAME_WE)
  string(REPLACE "@1" "${onnx_file}" cap "${CAPTURE_STR}")
  string(REPLACE "@2" "${fname}" cap "${cap}")
  string(APPEND ALL_CAPTURES "${cap}")
endforeach()

set(EMIT_CAPTURES "${ALL_CAPTURES}")

configure_file(EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)
configure_file(EmitFromONNX_GPU_ALPAKA.cxx.in EmitFromONNX_GPU_ALPAKA_all.cxx @ONLY)

############################################################################
# Low rank factorization tests (CPU) — builds an RModel programmatically
# (no ROOT interpreter, no ONNX needed) and validates Options::kLowRankFactorize.
# The GPU counterpart (alpaka/TestAlpakaLowRank.cxx) reuses the same emitted
# fixture; see the ENABLE_ALPAKA_TESTS block below.
############################################################################
ROOTTEST_GENERATE_EXECUTABLE(
  emitLowRankModels
  LowRankModelGenerator.cxx
  LIBRARIES SOFIE_core
  FIXTURES_SETUP sofie-lowrank-emit-build
)

ROOTTEST_ADD_TEST(
  EmitLowRankModels
  COMMAND ./emitLowRankModels
  FIXTURES_REQUIRED sofie-lowrank-emit-build
  FIXTURES_SETUP sofie-lowrank-emit
)

ROOTTEST_GENERATE_EXECUTABLE(
  TestLowRankFactorization
  TestLowRankFactorization.cxx
  LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main BLAS::BLAS
  FIXTURES_REQUIRED sofie-lowrank-emit
  FIXTURES_SETUP sofie-test-lowrank-build
)

target_include_directories(TestLowRankFactorization PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

ROOTTEST_ADD_TEST(
  TestLowRankFactorization
  EXEC ./TestLowRankFactorization
  FIXTURES_REQUIRED sofie-lowrank-emit
)

############################################################################
# Alpaka tests
############################################################################
if (ENABLE_ALPAKA_TESTS)

  string(TOLOWER "${ALPAKA_BACKEND}" _alpaka_backend)
  if (NOT _alpaka_backend IN_LIST ALPAKA_BACKEND)
    message(FATAL_ERROR "Unsupported ALPAKA_BACKEND=${ALPAKA_BACKEND}")
  endif()

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

  ##########################################################################
  # Alpaka emitter
  ##########################################################################
  ROOTTEST_GENERATE_EXECUTABLE(
    emitFromONNXAlpaka
    EmitFromONNX_GPU_ALPAKA_all.cxx
    LIBRARIES protobuf::libprotobuf SOFIE_core SOFIE_parsers
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka-build
  )

  target_compile_options(emitFromONNXAlpaka PRIVATE
    -Wno-unused-parameter
    -Wno-array-bounds
  )

  ROOTTEST_ADD_TEST(
    SofieCompileModels_ONNX_Alpaka
    COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNXAlpaka
    FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka-build
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka
  )

  ##########################################################################
  # CUDA backend
  ##########################################################################
  if (_alpaka_backend STREQUAL "cuda")

    message(STATUS "Enabling Alpaka CUDA tests")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(ALPAKA_CUDA_TEST_SOURCES
      alpaka/TestAlpakaLinear.cxx
      alpaka/TestAlpakaConv.cxx
      alpaka/TestAlpakaNormalization.cxx
      alpaka/TestAlpakaElementwiseUnary.cxx
      alpaka/TestAlpakaElementwiseBinaryLogic.cxx
      alpaka/TestAlpakaReduce.cxx
      alpaka/TestAlpakaShapeIndexing.cxx
      alpaka/TestAlpakaGNN.cxx
      alpaka/TestAlpakaPool.cxx
      alpaka/TestAlpakaLowRank.cxx
    )

    set_source_files_properties(
      ${ALPAKA_CUDA_TEST_SOURCES}
      PROPERTIES LANGUAGE CUDA
    )

    ROOTTEST_GENERATE_EXECUTABLE(
      TestCustomModelsFromONNXForAlpakaCuda
      ${ALPAKA_CUDA_TEST_SOURCES}
      LIBRARIES SOFIE_core GTest::gtest GTest::gtest_main
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit
      FIXTURES_SETUP sofie-test-models-onnx-alpaka-build
    )

    target_include_directories(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${alpaka_SOURCE_DIR}/include
      ${sofieblas_SOURCE_DIR}/include
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # No CUDA_ARCHITECTURES override here: let the target inherit whatever
    # CMAKE_CUDA_ARCHITECTURES was set globally (e.g. "native" from CI, so it
    # matches whatever GPU the runner actually has). A hardcoded list here
    # previously included sm_70 (Volta), which some CUDA toolkit installs no
    # longer support codegen for, breaking the build with "Unsupported gpu
    # architecture 'compute_70'" regardless of the actual runner's GPU.
    set_target_properties(
      TestCustomModelsFromONNXForAlpakaCuda
      PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    target_compile_definitions(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ALPAKA_ACC_GPU_CUDA_ENABLED
      ALPAKA_HAS_STD_ATOMIC_REF
    )

    target_compile_options(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --generate-line-info
        --use_fast_math
        -g
        -G
        # -fsanitize=address
        -O1
        -Wno-deprecated-gpu-targets
      >
      $<$<COMPILE_LANGUAGE:CXX>:
        -O2
        -g
        -G
        -fPIC
        -pthread
      >
    )
  # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

    # ROOT-compatible: plain signature only
    target_link_libraries(
      TestCustomModelsFromONNXForAlpakaCuda
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
    )

    ROOTTEST_ADD_TEST(
      TestCustomModelsFromONNXForAlpakaCuda
      EXEC ./TestCustomModelsFromONNXForAlpakaCuda
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit
    )

  elseif (_alpaka_backend STREQUAL "hip")
    message(FATAL_ERROR
      "ALPAKA_BACKEND=hip is not implemented yet (AMD GPU support is planned, "
      "see test/alpaka/). Use ALPAKA_BACKEND=cuda for now.")
  elseif (_alpaka_backend STREQUAL "sycl")
    message(FATAL_ERROR "ALPAKA_BACKEND=sycl is not implemented yet.")
  elseif (_alpaka_backend STREQUAL "cpu")
    message(FATAL_ERROR "ALPAKA_BACKEND=cpu is not implemented yet.")
  endif() # cuda backend
endif()   # ENABLE_ALPAKA_TESTS
