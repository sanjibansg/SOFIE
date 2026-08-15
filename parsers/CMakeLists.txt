# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building SOFIE package
############################################################################
#Author: Sitong An, Lorenzo Moneta 10/03/2021

#sofie is built only if protobuf is found

add_library(SOFIE_parsers SHARED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "onnx_proto3")
set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES GENERATED TRUE)
## silence protobuf warnings for version 3.6. Not needed from protobuf version 3.17
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-array-bounds)
## silence protobuf warnings for version 3.0. Not needed from protobuf version 3.6
set_source_files_properties(${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)


set(source_headers
   SOFIE/RModelParser_ONNX.hxx
)
list(TRANSFORM source_headers PREPEND "inc/")
target_include_directories(SOFIE_parsers
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set(sources_cxx
    src/RModelParser_ONNX.cxx
    src/ParseBasicUnary.cxx
    src/ParseBasicBinary.cxx
    src/ParseBasicIs.cxx
    src/ParseBatchNormalization.cxx
    src/ParseCast.cxx
    src/ParseConcat.cxx
    src/ParseConv.cxx
    src/ParseConvTranspose.cxx
    src/ParseGemm.cxx
    src/ParseGRU.cxx
    src/ParseIdentity.cxx
    src/ParseLeakyRelu.cxx
    src/ParseLSTM.cxx
    src/ParseBasicNary.cxx
    src/ParsePool.cxx
    src/ParseReduce.cxx
    src/ParseRelu.cxx
    src/ParseReshape.cxx
    src/ParseRNN.cxx
    src/ParseSelu.cxx
    src/ParseShape.cxx
    src/ParseSigmoid.cxx
    src/ParseSlice.cxx
    src/ParseSoftmax.cxx
    src/ParseTanh.cxx
    src/ParseTranspose.cxx
    src/ParseErf.cxx
    src/ParseRange.cxx
    src/ParseLayerNormalization.cxx
    src/ParseExpand.cxx
    src/ParseGather.cxx
    src/ParseGatherND.cxx
    src/ParseElu.cxx
    src/ParseFuseConvAdd.cxx
    src/ParseFuseConvTransposeAdd.cxx
    src/ParseFuseGemmRelu.cxx
    src/ParseFuseBatchnormRelu.cxx
    src/ParseFuseMatMulAdd.cxx
    src/ParseMatMul.cxx
    src/ParseComparision.cxx
    src/ParseEyeLike.cxx
    src/ParseConstant.cxx
    src/ParseTopK.cxx
    src/ParseTile.cxx
    src/ParseSplit.cxx
    src/ParseIf.cxx
    src/ParsePad.cxx
    src/ParseWhere.cxx
    src/ParseEinsum.cxx
    src/ParseRandom.cxx
    src/ParseNot.cxx
    src/ParseClip.cxx
    src/ParseScatterElements.cxx
    src/ParseTrilu.cxx
    src/ParseLogic.cxx
    src/ParseNonZero.cxx
    src/ParseScatterND.cxx
    ${PROTO_SRCS}
  ${DEPENDENCIES}
    ${SOFIE_core}
)
target_link_libraries(SOFIE_parsers PRIVATE SOFIE_core)
target_sources(SOFIE_parsers PRIVATE ${sources_headers} ${sources_cxx})

target_include_directories(SOFIE_parsers BEFORE PUBLIC
  ${Protobuf_INCLUDE_DIRS})
  target_link_libraries(SOFIE_parsers
  PUBLIC
    protobuf::libprotobuf
    ${SOFIE_core}
)
target_include_directories(SOFIE_parsers PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  ${Protobuf_INCLUDE_DIRS}
)

  set_target_properties(SOFIE_parsers PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE)

if(SOFIE_WITH_ROOT AND ROOT_FOUND)
  ROOT_GENERATE_DICTIONARY(G__SOFIE_parsers ${sources_headers}
    LINKDEF inc/LinkDef.h
    MODULE SOFIE_parsers
    OPTIONS --deep
  )
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_parsers_rdict.pcm
                ${CMAKE_CURRENT_BINARY_DIR}/libSOFIE_parsers.rootmap
          DESTINATION lib)
endif()

install(TARGETS SOFIE_parsers
  EXPORT SOFIETargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/inc/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
