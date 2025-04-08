#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_Conv.hxx"
#include "onnx_proto3.pb.h"


namespace SOFIE {

ParserFuseFuncSignature ParseFuseConvAdd = [](RModelParser_ONNX & /*parser*/, const onnx::NodeProto & /*convnode*/,
                                              const onnx::NodeProto & /*addnode*/) {
   std::unique_ptr<ROperator> op;
   return op;
};

} // namespace SOFIE
