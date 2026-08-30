#include "SOFIE/RModelParser_ONNX.hxx"
#include "SOFIE/ROperator_L2Normalization.hxx"
#include "onnx_proto3.pb.h"

namespace SOFIE {

    std::unique_ptr<ROperator> ParseFuseL2Normalization(RModelParser_ONNX &parser, const onnx::NodeProto &reduceNode,
                             const onnx::NodeProto &divNode, float epsilon)
    {
        const std::string inputName = reduceNode.input(0);
        const std::string outputName = divNode.output(0);

        if (!parser.IsRegisteredTensorType(inputName))
            throw std::runtime_error("SOFIE L2Normalization input type is not registered: " + inputName);

        const ETensorType inputType = parser.GetTensorType(inputName);
        std::unique_ptr<ROperator> op;

        switch (inputType) {
            case ETensorType::FLOAT:
                op = std::make_unique<ROperator_L2Normalization<float>>(inputName, epsilon, outputName);
                break;
            default:
                throw std::runtime_error("SOFIE L2Normalization currently supports only FLOAT input");
        }

        if (!parser.IsRegisteredTensorType(outputName))
            parser.RegisterTensorType(outputName, inputType);

        return op;
    }

} // namespace SOFIE