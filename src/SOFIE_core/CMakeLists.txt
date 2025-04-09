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

set(source_headers
   SOFIE/OperatorList.hxx
   SOFIE/RModel_Base.hxx
   SOFIE/RModel.hxx
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

list(TRANSFORM sources_headers PREPEND "inc/")

set(sources_cxx
    src/RModel_Base.cxx
    src/RModel.cxx
    src/RModel_GNN.cxx
    src/RModel_GraphIndependent.cxx
    src/RFunction.cxx
    src/RFunction_MLP.cxx
    src/RFunction_Mean.cxx
    src/RFunction_Sum.cxx
    src/SOFIE_common.cxx
)

target_sources(SOFIE_core PRIVATE ${sources_headers} ${sources_cxx})
target_include_directories(SOFIE_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(SOFIE_core PUBLIC
    Tree
)
# ROOT_GENERATE_DICTIONARY(G__SOFIE ${sources_headers}
#     LINKDEF inc/LinkDef1.h
#     MODULE SOFIE_core
#     OPTIONS --deep
# )

install(TARGETS SOFIE_core
        LIBRARY DESTINATION lib
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/" DESTINATION "include")

if(testing)
  add_subdirectory(test)
endif()
