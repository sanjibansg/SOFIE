# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building SOFIE standalone package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

#sofie is built only if protobuf is found

add_library(SOFIE_core SHARED)

set(sources_headers
   SOFIE/OperatorList.hxx
   SOFIE/RModel_Base.hxx
   SOFIE/RModel.hxx
   SOFIE/RModelExtension.hxx
   SOFIE/RModelCodegenPass.hxx
   SOFIE/RModelProfiler.hxx
   SOFIE/RModelProfilerGPU.hxx
   SOFIE/RWeightFile.hxx
   SOFIE/ROperator.hxx
   SOFIE/ROperator_BasicUnary.hxx
   SOFIE/ROperator_BasicBinary.hxx
   SOFIE/ROperator_BasicNary.hxx
   SOFIE/ROperator_BatchNormalization.hxx
   SOFIE/ROperator_Conv.hxx
   SOFIE/ROperator_ConvTranspose.hxx
   SOFIE/ROperator_Gemm.hxx
   SOFIE/ROperator_Relu.hxx
   SOFIE/ROperator_Tanh.hxx
   SOFIE/ROperator_LeakyRelu.hxx
   SOFIE/ROperator_Reshape.hxx
   SOFIE/ROperator_Selu.hxx
   SOFIE/ROperator_Sigmoid.hxx
   SOFIE/ROperator_Slice.hxx
   SOFIE/ROperator_Transpose.hxx
   SOFIE/ROperator_Pool.hxx
   SOFIE/ROperator_Concat.hxx
   SOFIE/ROperator_Identity.hxx
   SOFIE/ROperator_Constant.hxx
   SOFIE/ROperator_Softmax.hxx
   SOFIE/ROperator_Reduce.hxx
   SOFIE/ROperator_Shape.hxx
   SOFIE/ROperator_Cast.hxx
   SOFIE/ROperator_Custom.hxx
   SOFIE/ROperator_LayerNormalization.hxx
   SOFIE/ROperator_Expand.hxx
   SOFIE/ROperator_Erf.hxx
   SOFIE/ROperator_Swish.hxx
   SOFIE/ROperator_Elu.hxx
   SOFIE/ROperator_Comparision.hxx
   SOFIE/ROperator_EyeLike.hxx
   SOFIE/ROperator_Range.hxx
   SOFIE/ROperator_TopK.hxx
   SOFIE/ROperator_Tile.hxx
   SOFIE/ROperator_Split.hxx
   SOFIE/ROperator_SubGraph.hxx
   SOFIE/ROperator_Pad.hxx
   SOFIE/ROperator_QONNXQuant.hxx
   SOFIE/ROperator_ONNXQuantizeLinear.hxx
   SOFIE/ROperator_Where.hxx
   SOFIE/ROperator_Einsum.hxx
   SOFIE/ROperator_Random.hxx
   SOFIE/ROperator_ScatterElements.hxx
   SOFIE/SOFIE_common.hxx
   SOFIE/SOFIEHelpers.hxx

   SOFIE/FunctionList.hxx
   SOFIE/RModel_GNN.hxx
   SOFIE/RModel_GraphIndependent.hxx
   SOFIE/RFunction.hxx
   SOFIE/RFunction_MLP.hxx
   SOFIE/RFunction_Sum.hxx
   SOFIE/RFunction_Mean.hxx
)

# The quantization pipeline's headers. LinkDef names no quantization class, so keeping
# them out of the base list leaves the base dictionary complete.
set(sources_quantization_headers
   SOFIE/quantization/ROperator_QuantizedConv.hxx
   SOFIE/quantization/ROperator_QuantizedElementwise.hxx
   SOFIE/quantization/ROperator_QuantizedGather.hxx
   SOFIE/quantization/ROperator_QuantizedGemm.hxx
   SOFIE/quantization/ROperator_QuantizedMatMul.hxx
   SOFIE/quantization/ROperator_QuantizedMatrix.hxx
   SOFIE/SOFIE_Quantized.hxx
   SOFIE/SOFIE_QuantizedAlpaka.hxx
   SOFIE/quantization/RQuantization_AlpakaCommon.hxx
   SOFIE/quantization/RQuantization_AlpakaPrimitives.hxx
   SOFIE/quantization/RQuantization_AlpakaBroadcast.hxx
   SOFIE/quantization/RQuantization_AlpakaDenseLinear.hxx
   SOFIE/quantization/RQuantization_AlpakaDenseLinearCommon.hxx
   SOFIE/quantization/RQuantization_AlpakaDenseLinearInt8.hxx
   SOFIE/quantization/RQuantization_AlpakaDenseLinearFP8.hxx
   SOFIE/quantization/RQuantization_AlpakaConvolution.hxx
   SOFIE/quantization/RQuantization_AlpakaConvolutionCommon.hxx
   SOFIE/quantization/RQuantization_AlpakaConvolutionInt8.hxx
   SOFIE/quantization/RQuantization_AlpakaConvolutionFP8.hxx
   SOFIE/quantization/RQuantization_AlpakaElementwise.hxx
   SOFIE/quantization/RQuantization_AlpakaGather.hxx
   SOFIE/RQuantization.hxx
   SOFIE/quantization/RQuantization_Analysis.hxx
   SOFIE/quantization/RQuantization_Convolution.hxx
   SOFIE/quantization/RQuantization_ConvolutionTypes.hxx
   SOFIE/quantization/RQuantization_DenseLinear.hxx
   SOFIE/quantization/RQuantization_DenseLinearTypes.hxx
   SOFIE/quantization/RQuantization_Elementwise.hxx
   SOFIE/quantization/RQuantization_Gather.hxx
   SOFIE/quantization/RQuantization_Lowering.hxx
   SOFIE/quantization/RQuantization_Parameters.hxx
   SOFIE/quantization/RQuantization_Pipeline.hxx
   SOFIE/quantization/RQuantization_Storage.hxx
)

list(TRANSFORM sources_headers PREPEND "inc/")
list(TRANSFORM sources_quantization_headers PREPEND "inc/")

# The quantization pipeline's translation units. RModel_Quantization.cxx belongs here
# despite defining RModel members, because base calls none of them.
set(sources_quantization_cxx
    src/quantization/RModel_Quantization.cxx
    src/quantization/RQuantization_Analysis.cxx
    src/quantization/RQuantization_Convolution.cxx
    src/quantization/RQuantization_ConvolutionLowering.cxx
    src/quantization/RQuantization_DenseLinear.cxx
    src/quantization/RQuantization_DenseLinearPass.cxx
    src/quantization/RQuantization_Elementwise.cxx
    src/quantization/RQuantization_Gather.cxx
    src/quantization/RQuantization_Parameters.cxx
    src/quantization/RQuantization_Pipeline.cxx
    src/quantization/RQuantization_Storage.cxx
)

set(sources_cxx
    src/RModel_Base.cxx
    src/RModel.cxx
    src/RModelCodegenPass.cxx
    src/RWeightFile.cxx
    src/RModelProfiler.cxx
    src/RModelProfilerGPU.cxx
    src/RModel_ALPAKA.cxx
    src/RModel_GNN.cxx
    src/RModel_GraphIndependent.cxx
    src/RFunction.cxx
    src/RFunction_MLP.cxx
    src/RFunction_Mean.cxx
    src/RFunction_Sum.cxx
    src/SOFIE_common.cxx
)

target_sources(SOFIE_core PRIVATE ${sources_headers} ${sources_cxx})
target_include_directories(SOFIE_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(SOFIE_core PUBLIC utils)

if(SOFIE_WITH_ROOT AND ROOT_FOUND)
  target_compile_definitions(SOFIE_core PUBLIC SOFIE_SUPPORT_ROOT_BINARY)
  target_link_libraries(SOFIE_core PUBLIC Tree Core RIO)

  ROOT_GENERATE_DICTIONARY(G__SOFIE_core ${sources_headers}
      LINKDEF inc/LinkDef.h
      MODULE SOFIE_core
      OPTIONS --deep
  )

  # Install the dictionaries.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_core_rdict.pcm
                ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_core.rootmap
          DESTINATION lib)
endif()

# The quantization pipeline. Depends on base SOFIE, which depends on nothing here, so the
# link below runs one way.
add_library(SOFIE_quantization SHARED)
target_sources(SOFIE_quantization PRIVATE ${sources_quantization_headers} ${sources_quantization_cxx})
target_link_libraries(SOFIE_quantization PUBLIC SOFIE_core)

install(TARGETS SOFIE_core SOFIE_quantization
  EXPORT SOFIETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
