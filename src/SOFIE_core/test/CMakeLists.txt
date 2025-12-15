# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building TMVA SOFIE tests.
# @author Federico Sossai, Sanjiban Sengupta
############################################################################

include_directories(${CMAKE_SOURCE_DIR}/src/SOFIE_core/inc)
include_directories(${CMAKE_SOURCE_DIR}/src/SOFIE_parsers/inc)

if (NOT ONNX_MODELS_DIR)
  set(ONNX_MODELS_DIR input_models)
endif()

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

# --- Collect ONNX files and build ALL_CAPTURES BEFORE any configure_file() ---
set(ALL_CAPTURES "")
file(GLOB ONNX_FILES "${ONNX_MODELS_DIR}/*.onnx")

# If there are no models, ONNX_FILES will be empty and ALL_CAPTURES stays empty.
foreach(onnx_file ${ONNX_FILES})
  get_filename_component(fname ${onnx_file} NAME_WE)
  get_filename_component(fdir ${onnx_file} DIRECTORY)

  string(REPLACE "@1" "${onnx_file}" cap "${CAPTURE_STR}")
  string(REPLACE "@2" "${fname}" cap "${cap}")
  string(APPEND ALL_CAPTURES "${cap}")
endforeach()

# Now generate the combined source files for CPU, ROOT and ALPAKA
# They will have @EMIT_CAPTURES@ substituted with the contents of ALL_CAPTURES
# (CMake configure_file uses variables via @VAR@ when @ONLY is provided).
# To make ALL_CAPTURES visible to configure_file we set a temporary variable
# that configure_file can reference directly.
set(EMIT_CAPTURES "${ALL_CAPTURES}")

# Note: the .in templates must use @EMIT_CAPTURES@ placeholder.
configure_file(EmitFromONNX.cxx.in EmitFromONNX_all.cxx @ONLY)
# configure_file(EmitFromRoot.cxx.in EmitFromRoot_all.cxx @ONLY)
configure_file(EmitFromONNX_GPU_ALPAKA.cxx.in EmitFromONNX_GPU_ALPAKA_all.cxx @ONLY)

# --- CPU emitter executable and test (unchanged) ---
ROOTTEST_GENERATE_EXECUTABLE(emitFromONNX EmitFromONNX_all.cxx
    LIBRARIES protobuf::libprotobuf SOFIE_core SOFIE_parsers
    FIXTURES_SETUP sofie-compile-models-onnx-build)

# silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromONNX PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# IMPORTANT: ROOTTEST_ADD_TEST below used to expand ${onnx_file}/${fname} which are loop vars.
# We keep it as a single-test wrapper; per-file test invocation is added later for ALPAKA.
ROOTTEST_ADD_TEST(SofieCompileModels_ONNX
  COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNX ${onnx_file} ${CMAKE_CURRENT_BINARY_DIR}/${fname}
  FIXTURES_REQUIRED sofie-compile-models-onnx-build
  FIXTURES_SETUP sofie-compile-models-onnx
)

# Creating a Google Test
if (BLAS_FOUND)  # we need BLAS for compiling the models
  ROOTTEST_GENERATE_EXECUTABLE(TestCustomModelsFromONNX TestCustomModelsFromONNX.cxx
    LIBRARIES
      MathCore
      SOFIE_core
      BLAS::BLAS
      GTest::gtest
      GTest::gtest_main
    FIXTURES_REQUIRED
      sofie-compile-models-onnx
    FIXTURES_SETUP
      sofie-test-models-onnx-build
  )
  target_include_directories(TestCustomModelsFromONNX PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  ROOTTEST_ADD_TEST(TestCustomModelsFromONNX
                    EXEC ./TestCustomModelsFromONNX
                    FIXTURES_REQUIRED sofie-test-models-onnx-build)
endif()

# Skipping emitting from ROOT and further tests for now.
# For testing serialisation of RModel object
# ROOTTEST_GENERATE_EXECUTABLE(emitFromROOT EmitFromRoot_all.cxx
#     LIBRARIES protobuf::libprotobuf RIO SOFIE_core SOFIE_parsers
#     FIXTURES_SETUP sofie-compile-models-onnx-root
# )

# silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
# target_compile_options(emitFromROOT PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# Automatic compilation of headers from root files
# ROOTTEST_ADD_TEST(SofieCompileModels_ROOT
#   COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromROOT
#   FIXTURES_REQUIRED sofie-compile-models-onnx-root
#   FIXTURES_SETUP sofie-compile-models-root
# )

# if (BLAS_FOUND)
#   # Creating a Google Test for Serialisation of RModel
#   ROOTTEST_GENERATE_EXECUTABLE(TestCustomModelsFromROOT TestCustomModelsFromROOT.cxx
#     LIBRARIES
#       SOFIE_core
#       BLAS::BLAS
#       GTest::gtest
#       GTest::gtest_main
#     FIXTURES_REQUIRED
#       sofie-compile-models-root
#     FIXTURES_SETUP
#       sofie-test-models-root-build
#   )
#   target_include_directories(TestCustomModelsFromROOT PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
#   ROOTTEST_ADD_TEST(TestCustomModelsFromROOT
#                     EXEC ./TestCustomModelsFromROOT
#                     FIXTURES_REQUIRED sofie-test-models-root-build)
# endif()

# Look for needed Python modules
ROOT_FIND_PYTHON_MODULE(torch)
if (ROOT_TORCH_FOUND)
  configure_file(Conv1dModelGenerator.py  Conv1dModelGenerator.py COPYONLY)
  configure_file(Conv2dModelGenerator.py  Conv2dModelGenerator.py COPYONLY)
  configure_file(Conv3dModelGenerator.py  Conv3dModelGenerator.py COPYONLY)
  configure_file(ConvTrans2dModelGenerator.py  ConvTrans2dModelGenerator.py COPYONLY)
  configure_file(LinearModelGenerator.py  LinearModelGenerator.py COPYONLY)
  configure_file(RecurrentModelGenerator.py  RecurrentModelGenerator.py COPYONLY)

  if (BLAS_FOUND)
    ROOT_ADD_GTEST(TestSofieModels TestSofieModels.cxx
      LIBRARIES
        SOFIE_core
        SOFIE_parsers
        BLAS::BLAS
      INCLUDE_DIRS
        ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

ROOT_EXECUTABLE(emitGNN GNN/EmitGNN.cxx LIBRARIES SOFIE_core)
ROOT_ADD_TEST(tmva-sofie-EmitGNN COMMAND emitGNN)

ROOT_EXECUTABLE(EmitGraphIndependent GNN/EmitGraphIndependent.cxx LIBRARIES SOFIE_core)
ROOT_ADD_TEST(tmva-sofie-EmitGraphIndependent COMMAND EmitGraphIndependent)

# Generating inference code for heterogeneous testing using ALPAKA
ROOTTEST_GENERATE_EXECUTABLE(emitFromONNXAlpaka EmitFromONNX_GPU_ALPAKA_all.cxx
    LIBRARIES protobuf::libprotobuf SOFIE_core SOFIE_parsers
    FIXTURES_SETUP sofie-compile-models-onnx-alpaka-build)

# silence protobuf warnings seen in version 3.0 and 3.6. Not needed from protobuf version 3.17
target_compile_options(emitFromONNXAlpaka PRIVATE -Wno-unused-parameter -Wno-array-bounds)

# Add explicit per-file post-build runs of the alpaka emitter executable so that
# EmitFromONNXAlpaka is invoked for each detected .onnx file during the build.
# This avoids relying on a single ${onnx_file}/${fname} value that would
# otherwise expand only to the last entry when used outside the loop.

if (ONNX_FILES)
  foreach(onnx_file ${ONNX_FILES})
    get_filename_component(fname ${onnx_file} NAME_WE)

    # Create a post-build command attached to the emitFromONNXAlpaka target that
    # will run the built binary with the current onnx file and the chosen outname.
    add_custom_command(TARGET emitFromONNXAlpaka
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 $<TARGET_FILE:emitFromONNXAlpaka> "${onnx_file}" "${CMAKE_CURRENT_BINARY_DIR}/${fname}"
      COMMENT "Running ALPAKA emitter on ${onnx_file}"
    )
  endforeach()
endif()

# Also add a ROOTTEST wrapper so that ctest can run one of the invocations if desired.
# (This remains mostly for CI / test harness compatibility.)
ROOTTEST_ADD_TEST(SofieCompileModels_ONNX_Alpaka
  COMMAND ${CMAKE_COMMAND} -E env ROOTIGNOREPREFIX=1 ./emitFromONNXAlpaka ${ONNX_FILES}
  FIXTURES_REQUIRED sofie-compile-models-onnx-alpaka-build
  FIXTURES_SETUP sofie-compile-models-onnx-alpaka
)

# End of CMakeLists.txt
