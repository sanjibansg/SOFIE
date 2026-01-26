# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE tests.
# @author Federico Sossai, Sanjiban Sengupta
############################################################################

cmake_minimum_required(VERSION 3.14)
include(FetchContent)

include_directories(${CMAKE_SOURCE_DIR}/src/SOFIE_core/inc)
include_directories(${CMAKE_SOURCE_DIR}/src/SOFIE_parsers/inc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT ONNX_MODELS_DIR)
  set(ONNX_MODELS_DIR input_models)
endif()

option(ENABLE_ALPAKA_TESTS "Enable Alpaka-based SOFIE tests" OFF)
set(ALPAKA_BACKEND "cuda"
    CACHE STRING "Alpaka backend to test (cuda, cpu, hip, sycl)")
set_property(CACHE ALPAKA_BACKEND PROPERTY STRINGS cuda cpu hip sycl)

# String template used to produce calls to EmitModel(...) per file.
set(CAPTURE_STR
"try {\n\
    EmitModel(\"@1\", \"@2\");\n\
} catch (const std::exception& e) {\n\
    std::string msg = e.what();\n\
    if (msg.find(\"multiple output tensors are not supported\") != std::string::npos) {\n\
        std::cerr << \"[SKIP] Multiple outputs are not supported for @1\" << std::endl;\n\
    } else {\n\
        std::cerr << \"[ERROR] Failed processing @1: \" << msg << std::endl;\n\
        failures++;\n\
    }\n\
} catch (...) {\n\
    std::cerr << \"[ERROR] Unknown failure processing @1\" << std::endl;\n\
    failures++;\n\
}\n\
")

# --- Collect ONNX files ---
set(ALL_CAPTURES "")
file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")

foreach(onnx_file ${ONNX_FILES})
  get_filename_component(fname ${onnx_file} NAME_WE)
  string(REPLACE "@1" "${onnx_file}" cap "${CAPTURE_STR}")
  string(REPLACE "@2" "${fname}" cap "${cap}")
  string(APPEND ALL_CAPTURES "${cap}")
endforeach()

set(EMIT_CAPTURES "${ALL_CAPTURES}")

configure_file(EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)
configure_file(EmitFromONNX_GPU_ALPAKA.cxx.in EmitFromONNX_GPU_ALPAKA_all.cxx @ONLY)

# --- CPU emitter ---
# ROOTTEST_GENERATE_EXECUTABLE(emitFromONNX EmitFromONNX_all.cxx
#   LIBRARIES protobuf::libprotobuf SOFIE_core SOFIE_parsers
#   FIXTURES_SETUP sofie-compile-models-onnx-build)

# target_compile_options(emitFromONNX PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# ROOTTEST_ADD_TEST(SofieCompileModels_ONNX
#   COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX
#   FIXTURES_REQUIRED sofie-compile-models-onnx-build
#   FIXTURES_SETUP sofie-compile-models-onnx
# )

# # --- Custom model tests ---
# if (BLAS_FOUND)
#   ROOTTEST_GENERATE_EXECUTABLE(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
#     LIBRARIES
#       MathCore
#       SOFIE_core
#       BLAS::BLAS
#       GTest::gtest
#       GTest::gtest_main
#     FIXTURES_REQUIRED sofie-compile-models-onnx
#     FIXTURES_SETUP sofie-test-models-onnx-build
#   )

#   target_include_directories(TestCustomModelsFromONNX PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

#   ROOTTEST_ADD_TEST(TestCustomModelsFromONNX
#     EXEC ./TestCustomModelsFromONNX
#     FIXTURES_REQUIRED sofie-test-models-onnx-build
#   )
# endif()

# # --- Python-based generators ---
# ROOT_FIND_PYTHON_MODULE(torch)
# if (ROOT_TORCH_FOUND)
#   configure_file(Conv1dModelGenerator.py  Conv1dModelGenerator.py COPYONLY)
#   configure_file(Conv2dModelGenerator.py  Conv2dModelGenerator.py COPYONLY)
#   configure_file(Conv3dModelGenerator.py  Conv3dModelGenerator.py COPYONLY)
#   configure_file(ConvTrans2dModelGenerator.py ConvTrans2dModelGenerator.py COPYONLY)
#   configure_file(LinearModelGenerator.py LinearModelGenerator.py COPYONLY)
#   configure_file(RecurrentModelGenerator.py RecurrentModelGenerator.py COPYONLY)

#   if (BLAS_FOUND)
#     ROOT_ADD_GTEST(TestSofieModels TestSofieModels.cxx
#       LIBRARIES SOFIE_core SOFIE_parsers BLAS::BLAS
#       INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
#     )
#   endif()
# endif()

# ROOT_EXECUTABLE(emitGNN GNN/EmitGNN.cxx LIBRARIES SOFIE_core)
# ROOT_ADD_TEST(tmva-sofie-EmitGNN COMMAND emitGNN)

# ROOT_EXECUTABLE(EmitGraphIndependent GNN/EmitGraphIndependent.cxx LIBRARIES SOFIE_core)
# ROOT_ADD_TEST(tmva-sofie-EmitGraphIndependent COMMAND EmitGraphIndependent)

# =========================
# ALPAKA TESTS
# =========================
if (ENABLE_ALPAKA_TESTS)

  string(TOLOWER "${ALPAKA_BACKEND}" _alpaka_backend)

  if (NOT _alpaka_backend IN_LIST ALPAKA_BACKEND)
    message(FATAL_ERROR
      "Unsupported ALPAKA_BACKEND='${ALPAKA_BACKEND}'. "
      "Valid values: cuda, cpu, hip, sycl")
  endif()

  FetchContent_Declare(
    sofieBLAS
    GIT_REPOSITORY https://github.com/ML4EP/sofieBLAS
    GIT_TAG        edf2259876e9f4fb5a8f72db20b2dfb5dc26b517
  )
  FetchContent_MakeAvailable(sofieBLAS)

  FetchContent_Declare(
    alpaka
    GIT_REPOSITORY https://github.com/alpaka-group/alpaka
    GIT_TAG        2fa91a34ed11b2076e474c5507d920e85cf9b79d
  )
  FetchContent_MakeAvailable(alpaka)

  # --- ALPAKA emitter ---
  ROOTTEST_GENERATE_EXECUTABLE(emitFromONNXAlpaka EmitFromONNX_GPU_ALPAKA_all.cxx
    LIBRARIES protobuf::libprotobuf SOFIE_core SOFIE_parsers
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka-build
  )

  target_compile_options(emitFromONNXAlpaka PRIVATE -Wno-unused-parameter -Wno-array-bounds)

  if (ONNX_FILES)
    foreach(onnx_file ${ONNX_FILES})
      get_filename_component(fname ${onnx_file} NAME_WE)
      add_custom_command(TARGET emitFromONNXAlpaka POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1
                $<TARGET_FILE:emitFromONNXAlpaka>
                "${onnx_file}" "${CMAKE_CURRENT_BINARY_DIR}/${fname}"
        COMMENT "Running ALPAKA emitter on ${onnx_file}")
    endforeach()
  endif()

  ROOTTEST_ADD_TEST(SofieCompileModels_ONNX_Alpaka
    COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNXAlpaka
    FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka-build
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka
  )

  set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
  set(CXX_HOST_FLAGS -fPIC -pthread)

  # ---- Backend selection ----
  if (_alpaka_backend STREQUAL "cuda")
    message(STATUS "Enabling Alpaka CUDA tests")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CUDA_ARCH "sm_86")
    set(CXX_CUDA_FLAGS
      -arch=${CUDA_ARCH}
      -Wno-deprecated-gpu-targets
      --extended-lambda
      --expt-relaxed-constexpr)

    set_source_files_properties(
      TestCustomModelsFromONNXForAlpakaCuda.cxx
      PROPERTIES LANGUAGE CUDA
    )


ROOTTEST_GENERATE_EXECUTABLE(
  TestCustomModelsFromONNXForAlpakaCuda
  TestCustomModelsFromONNXForAlpakaCuda.cxx
  LIBRARIES MathCore SOFIE_core GTest::gtest GTest::gtest_main
  FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka
  FIXTURES_SETUP sofie-test-models-onnx-alpaka-build
)


    target_include_directories(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${alpaka_SOURCE_DIR}/include
      ${SOFIE_INCLUDE}
      ${sofieblas_SOURCE_DIR}/include 
      ${ROOT_INCLUDE_DIRS}
      ${CUDA_BASE}/include
      ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_target_properties(
      TestCustomModelsFromONNXForAlpakaCuda
      PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_definitions(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ALPAKA_ACC_GPU_CUDA_ENABLED
    )

    target_link_directories(
      TestCustomModelsFromONNXForAlpakaCuda PRIVATE
      ${CUDA_BASE}/lib64
    )

    target_link_libraries(TestCustomModelsFromONNXForAlpakaCuda      
      CUDA::cublas
      CUDA::cublasLt
      CUDA::cudart
      ${ROOT_LIBRARIES}
    )

    ROOTTEST_ADD_TEST(TestCustomModelsFromONNXForAlpakaCuda
      EXEC ./TestCustomModelsFromONNXForAlpakaCuda
      FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka
    )

  elseif (_alpaka_backend STREQUAL "cpu")
    message(STATUS "Alpaka CPU backend selected (not yet implemented)")
  elseif (_alpaka_backend STREQUAL "hip")
    message(STATUS "Alpaka HIP backend selected (not yet implemented)")
  elseif (_alpaka_backend STREQUAL "sycl")
    message(STATUS "Alpaka SYCL backend selected (not yet implemented)")
  endif()  # backend

endif()  # ENABLE_ALPAKA_TESTS
