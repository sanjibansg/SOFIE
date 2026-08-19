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

############################################################################
# Package boundary
############################################################################
# Linking against SOFIE_core alone fails as soon as base references a symbol the
# quantization package defines, and running it exercises the no-pass-installed path.
add_executable(SofieCoreStandsAlone CoreStandsAlone.cxx)
target_link_libraries(SofieCoreStandsAlone PRIVATE SOFIE_core)
add_test(NAME SofieCoreStandsAlone COMMAND SofieCoreStandsAlone)

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
  FIXTURES_REQUIRED sofie-lowrank-emit;sofie-test-lowrank-build
)

############################################################################
# Alpaka tests
############################################################################
if (ENABLE_ALPAKA_TESTS)

  # sm_90 gives the FP8 paths native E4M3 instructions; below sm_89 nvcc falls back to software
  # emulation. Each listed architecture is a full compile pass.
  set(SOFIE_TEST_CUDA_ARCHITECTURES "70;80;86;90" CACHE STRING
      "CUDA architectures for the Alpaka/CUDA test targets")

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
    LIBRARIES protobuf::libprotobuf SOFIE_quantization SOFIE_parsers
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

  ROOTTEST_ADD_TEST(
    SofieQuantizedStorageExternalized_ONNX_Alpaka
    COMMAND ${CMAKE_COMMAND}
            -DGENERATED_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/CheckQuantizedStorageExternalized.cmake
    FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka
  )

  ##########################################################################
  # CUDA backend
  ##########################################################################
  if (_alpaka_backend STREQUAL "cuda")

    message(STATUS "Enabling Alpaka CUDA tests")

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(_alpaka_cuda_test_targets
      TestCustomModelsFromONNXForAlpakaCuda
      TestQuantizationFromONNXForAlpakaCuda
    )

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
      alpaka/TestAlpakaKernelOnly.cxx
      alpaka/TestAlpakaNormReduction.cxx
      alpaka/TestAlpakaSequenceOps.cxx
    )

    set_source_files_properties(
      ${ALPAKA_CUDA_TEST_SOURCES}
      TestQuantizationFromONNXForAlpakaCuda.cxx
      PROPERTIES LANGUAGE CUDA
    )

    ROOTTEST_GENERATE_EXECUTABLE(
      TestCustomModelsFromONNXForAlpakaCuda
      ${ALPAKA_CUDA_TEST_SOURCES}
      LIBRARIES SOFIE_quantization GTest::gtest GTest::gtest_main
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit
      FIXTURES_SETUP sofie-test-models-onnx-alpaka-build
    )

    ROOTTEST_GENERATE_EXECUTABLE(
      TestQuantizationFromONNXForAlpakaCuda
      TestQuantizationFromONNXForAlpakaCuda.cxx
      LIBRARIES SOFIE_quantization GTest::gtest GTest::gtest_main
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka
      FIXTURES_SETUP sofie-quantization-models-onnx-alpaka-build
    )

    foreach(_alpaka_cuda_test_target IN LISTS _alpaka_cuda_test_targets)
      target_include_directories(
        ${_alpaka_cuda_test_target} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${alpaka_SOURCE_DIR}/include
        ${sofieblas_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
      )

      set_target_properties(
        ${_alpaka_cuda_test_target}
        PROPERTIES
          CUDA_SEPARABLE_COMPILATION OFF
          CUDA_ARCHITECTURES "${SOFIE_TEST_CUDA_ARCHITECTURES}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
      )

      target_compile_definitions(
        ${_alpaka_cuda_test_target} PRIVATE
        ALPAKA_ACC_GPU_CUDA_ENABLED
        ALPAKA_HAS_STD_ATOMIC_REF
        SOFIE_USE_CUBLASLT
      )

      target_compile_options(
        ${_alpaka_cuda_test_target} PRIVATE
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

      # ROOT-compatible: plain signature only.
      target_link_libraries(
        ${_alpaka_cuda_test_target}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
      )
    endforeach()

    ROOTTEST_ADD_TEST(
      TestCustomModelsFromONNXForAlpakaCuda
      EXEC ./TestCustomModelsFromONNXForAlpakaCuda
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka;sofie-lowrank-emit;sofie-test-models-onnx-alpaka-build
    )

    ROOTTEST_ADD_TEST(
      TestQuantizationFromONNXForAlpakaCuda
      EXEC ./TestQuantizationFromONNXForAlpakaCuda
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka
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
