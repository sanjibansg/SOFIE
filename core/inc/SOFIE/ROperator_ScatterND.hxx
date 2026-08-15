#ifndef SOFIE_ROPERATOR_SCATTERND
#define SOFIE_ROPERATOR_SCATTERND

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SOFIE {

template <typename T>
class ROperator_ScatterND final : public ROperator
{
private:
   std::string fNData;
   std::string fNIndices;
   std::string fNUpdates;
   std::string fNY;
   std::string fReduction;

   std::vector<Dim> fShapeData;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeUpdates;
   std::vector<Dim> fShapeY;

   bool fIndicesDynamic = false;
   bool fUpdatesDynamic = false;

public:
   ROperator_ScatterND() {}

   ROperator_ScatterND(std::string nameData, std::string nameIndices, std::string nameUpdates, std::string nameY, std::string reduction)
      : fNData(UTILITY::Clean_name(nameData)), fNIndices(UTILITY::Clean_name(nameIndices)),
        fNUpdates(UTILITY::Clean_name(nameUpdates)), fNY(UTILITY::Clean_name(nameY)), fReduction(reduction)
   {
      fInputTensorNames = {fNData, fNIndices, fNUpdates};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return {input[0]};
   }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNData))
         throw std::runtime_error("SOFIE ScatterND data tensor " + fNData + " is not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNIndices))
         throw std::runtime_error("SOFIE ScatterND indices tensor " + fNIndices + " is not found in model");
      if (!model.CheckIfTensorAlreadyExist(fNUpdates))
         throw std::runtime_error("SOFIE ScatterND updates tensor " + fNUpdates + " is not found in model");

      if (model.GetTensorType(fNIndices) != ETensorType::INT64)
         throw std::runtime_error("SOFIE ScatterND indices must be INT64");

      fShapeData = model.GetDimTensorShape(fNData);
      fShapeIndices = model.GetDimTensorShape(fNIndices);
      fShapeUpdates = model.GetDimTensorShape(fNUpdates);

      if (fShapeData.empty() || fShapeIndices.empty())
         throw std::runtime_error("SOFIE ScatterND data and indices must have rank >= 1");

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();

      if (fShapeIndices.back().isParam)
         throw std::runtime_error("SOFIE ScatterND last indices dimension must be known");

      size_t k = fShapeIndices.back().dim;
      if (k > r)
         throw std::runtime_error("SOFIE ScatterND indices_shape[-1] must be <= rank(data)");

      if (fShapeUpdates.size() != q + r - k - 1)
         throw std::runtime_error("SOFIE ScatterND updates tensor has invalid rank");

      fShapeY = fShapeData;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNData), fShapeY);

      fIndicesDynamic = model.IsDynamicTensor(fNIndices);
      fUpdatesDynamic = model.IsDynamicTensor(fNUpdates);

      if (fReduction.empty())
         fReduction = "none";

      if (fReduction != "none" && fReduction != "add" && fReduction != "mul" && fReduction != "max" && fReduction != "min")
         throw std::runtime_error("SOFIE ScatterND unsupported reduction " + fReduction);
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;

      size_t r = fShapeData.size();
      size_t q = fShapeIndices.size();
      size_t k = fShapeIndices.back().dim;

      auto dataStrides = UTILITY::ComputeStrideFromShape(fShapeData);
      std::vector<Dim> updateSliceShape(fShapeData.begin() + k, fShapeData.end());
      std::string numTuples = ConvertDimShapeToLength(std::vector<Dim>(fShapeIndices.begin(), fShapeIndices.end() - 1));
      std::string sliceSize = ConvertDimShapeToLength(updateSliceShape);
      std::string dataLength = ConvertDimShapeToLength(fShapeData);

      std::stringstream out;
      out << "\n//------ ScatterND " << opName << "\n";
      out << SP << "std::copy(tensor_" << fNData << ", tensor_" << fNData << " + " << dataLength << ", tensor_" << fNY << ");\n";
      out << SP << "for (size_t tuple = 0; tuple < " << numTuples << "; ++tuple) {\n";
      out << SP << SP << "size_t outputBase = 0;\n";

      for (size_t d = 0; d < k; ++d) {
         out << SP << SP << "int64_t idx_" << d << " = tensor_" << fNIndices << "[tuple * " << k << " + " << d << "];\n";
         out << SP << SP << "if (idx_" << d << " < 0) idx_" << d << " += " << fShapeData[d].GetVal() << ";\n";
         out << SP << SP << "outputBase += static_cast<size_t>(idx_" << d << ") * " << dataStrides[d].GetVal() << ";\n";
      }

      out << SP << SP << "for (size_t s = 0; s < " << sliceSize << "; ++s) {\n";

      if (fReduction == "none")
         out << SP << SP << SP << "tensor_" << fNY << "[outputBase + s] = tensor_" << fNUpdates << "[tuple * " << sliceSize << " + s];\n";
      else if (fReduction == "add")
         out << SP << SP << SP << "tensor_" << fNY << "[outputBase + s] += tensor_" << fNUpdates << "[tuple * " << sliceSize << " + s];\n";
      else if (fReduction == "mul")
         out << SP << SP << SP << "tensor_" << fNY << "[outputBase + s] *= tensor_" << fNUpdates << "[tuple * " << sliceSize << " + s];\n";
      else if (fReduction == "max")
         out << SP << SP << SP << "tensor_" << fNY << "[outputBase + s] = std::max(tensor_" << fNY << "[outputBase + s], tensor_" << fNUpdates << "[tuple * " << sliceSize << " + s]);\n";
      else if (fReduction == "min")
         out << SP << SP << SP << "tensor_" << fNY << "[outputBase + s] = std::min(tensor_" << fNY << "[outputBase + s], tensor_" << fNUpdates << "[tuple * " << sliceSize << " + s]);\n";

      out << SP << SP << "}\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;

      size_t k = fShapeIndices.back().dim;
      auto dataStrides = UTILITY::ComputeStrideFromShape(fShapeData);
      std::vector<Dim> updateSliceShape(fShapeData.begin() + k, fShapeData.end());
      std::string sliceSize = ConvertDimShapeToLength(updateSliceShape);

      std::string op;
      op += "\n//------ SCATTERND_KERNEL_ALPAKA\n";
      op += SP + "struct ScatterNDKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, int64_t const* indices, T const* updates, T* output, std::size_t const numUpdates) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= numUpdates) return;\n";
      op += SP + SP + SP + "std::size_t const tuple = idx / " + sliceSize + ";\n";
      op += SP + SP + SP + "std::size_t const sliceOffset = idx % " + sliceSize + ";\n";
      op += SP + SP + SP + "std::size_t outputBase = 0;\n";

      for (size_t d = 0; d < k; ++d) {
         op += SP + SP + SP + "int64_t idx_" + std::to_string(d) + " = indices[tuple * " + std::to_string(k) + "u + " + std::to_string(d) + "u];\n";
         op += SP + SP + SP + "if (idx_" + std::to_string(d) + " < 0) idx_" + std::to_string(d) + " += " + fShapeData[d].GetVal() + ";\n";
         op += SP + SP + SP + "outputBase += static_cast<std::size_t>(idx_" + std::to_string(d) + ") * " + dataStrides[d].GetVal() + ";\n";
      }

      if (fReduction == "none")
         op += SP + SP + SP + "output[outputBase + sliceOffset] = updates[idx];\n";
      else if (fReduction == "add")
         op += SP + SP + SP + "alpaka::atomicAdd(acc, &output[outputBase + sliceOffset], updates[idx]);\n";
      else if (fReduction == "mul")
         op += SP + SP + SP + "alpaka::atomicMul(acc, &output[outputBase + sliceOffset], updates[idx]);\n";
      else if (fReduction == "max")
         op += SP + SP + SP + "alpaka::atomicMax(acc, &output[outputBase + sliceOffset], updates[idx]);\n";
      else if (fReduction == "min")
         op += SP + SP + SP + "alpaka::atomicMin(acc, &output[outputBase + sliceOffset], updates[idx]);\n";

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      return SP + "ScatterNDKernel_op_" + opName + " scatterNDKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      std::string indicesBuffer = (fIndicesDynamic ? "bufDev_" : "deviceBuf_") + fNIndices;
      std::string updatesBuffer = (fUpdatesDynamic ? "bufDev_" : "deviceBuf_") + fNUpdates;
      std::string dataLength = ConvertDimShapeToLength(fShapeData);
      std::vector<Dim> updateCountShape = fShapeUpdates;
      std::string numUpdates = ConvertDimShapeToLength(updateCountShape);

      std::stringstream out;
      out << "\n//------ ScatterND_GPU_ALPAKA\n";
      out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY << ", deviceBuf_" << fNData << ");\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(Vec::all(Idx{" << numUpdates << "}));\n";
      out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY << ", scatterNDKernel_" << opName
          << ", alpaka::getPtrNative(" << indicesBuffer << "), alpaka::getPtrNative(" << updatesBuffer
          << "), alpaka::getPtrNative(deviceBuf_" << fNY << "), static_cast<Idx>(" << numUpdates << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_SCATTERND
