#ifndef SOFIE_ROPERATOR_BASICNARY
#define SOFIE_ROPERATOR_BASICNARY

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <vector>
#include <sstream>
#include <algorithm>

namespace SOFIE{

enum class EBasicNaryOperator {Max, Min, Mean, Sum};

template<typename T, EBasicNaryOperator Op>
struct NaryOperatorTraits {};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Max> {
   static const std::string Name() {return "Max";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0] << ";\n";
      for (size_t i = 1; i < inputs.size(); i++) {
         out << "\t" << "\t" << res << " = std::max(" << res << ", " << inputs[i] << ");\n";
      }
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Min> {
   static const std::string Name() {return "Min";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0] << ";\n";
      for (size_t i = 1; i < inputs.size(); i++) {
         out << "\t" << "\t" << res << " = std::min(" << res << ", " << inputs[i] << ");\n";
      }
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Mean> {};

template<>
struct NaryOperatorTraits<float, EBasicNaryOperator::Mean> {
   static const std::string Name() {return "Mean";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = (" << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ") / float(" << inputs.size() << ");\n";
      return out.str();
   }
};

template<typename T>
struct NaryOperatorTraits<T, EBasicNaryOperator::Sum> {
   static const std::string Name() {return "Sum";}
   static std::string Op(const std::string& res, std::vector<std::string>& inputs) {
      std::stringstream out;
      out << "\t" << "\t" << res << " = " << inputs[0];
      for (size_t i = 1; i < inputs.size(); i++) {
         out << " + " << inputs[i];
      }
      out << ";\n";
      return out.str();
   }
};

template <typename T, EBasicNaryOperator Op>
class ROperator_BasicNary final : public ROperator
{

private:

   std::vector<std::string> fNInputs;
   std::string fNY;
   std::vector<std::vector<Dim>> fShapeInputs;

   std::vector<std::string> fNBroadcastedInputs;
   std::vector<Dim> fShapeY;

   bool fBroadcast = false;

   std::string fType;

public:
   ROperator_BasicNary(){}

   ROperator_BasicNary( const std::vector<std::string> & inputNames, const std::string& nameY):
   fNY(UTILITY::Clean_name(nameY)){
      fNInputs.reserve(inputNames.size());
      for (auto & name : inputNames)
         fNInputs.push_back(UTILITY::Clean_name(name));

      fInputTensorNames.resize(fNInputs.size());
      std::transform(fNInputs.begin(), fNInputs.end(), fInputTensorNames.begin(),
                  [](const std::string& s) -> std::string_view { return s; });
      fOutputTensorNames = { fNY };
   }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = std::vector<std::vector<size_t>>(1, input[0]);
      return ret;
   }

   void Initialize(RModel& model) override {
      for (auto &it : fNInputs) {
         if (!model.CheckIfTensorAlreadyExist(it)) {
            throw std::runtime_error("SOFIE BasicNary Op Input Tensor " + it + " is not found in model");
         }
         fShapeInputs.push_back(model.GetDimTensorShape(it));
      }
      // Find the common output shape by pairwise multidirectional broadcast
      fShapeY = fShapeInputs[0];
      for (size_t i = 1; i < fShapeInputs.size(); i++) {
         auto shapeA = fShapeY;
         auto shapeB = fShapeInputs[i];
         auto ret = UTILITY::MultidirectionalBroadcastShape(shapeA, shapeB);
         fShapeY = ret.second;
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNInputs[0]), fShapeY);
      // Broadcasting
      size_t N = fNInputs.size();
      fNBroadcastedInputs.reserve(N);
      for (size_t i = 0; i < N; i++) {
         if (!UTILITY::AreSameShape(fShapeInputs[i], fShapeY)) {
            fBroadcast = true;
            std::string name = "Broadcasted"  + fNInputs[i];
            model.AddIntermediateTensor(name, model.GetTensorType(fNInputs[0]), fShapeY);
            fNBroadcastedInputs.emplace_back("tensor_" + name);
         } else {
            fNBroadcastedInputs.emplace_back("tensor_" + fNInputs[i]);
         }
      }
      fType = ConvertTypeToString(model.GetTensorType(fNInputs[0]));
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeY.empty()) {
         throw std::runtime_error("SOFIE BasicNary called to Generate without being initialized first");
      }
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShapeY);
      out << SP << "\n//------ BasicNary operator\n";
      if (fBroadcast) {
         for (size_t i = 0; i < fNInputs.size(); i++) {
            if (fNBroadcastedInputs[i] != fNInputs[i]) {
               out << SP << SP << "// Broadcasting " << fNInputs[i] << " to " << ConvertDimShapeToString(fShapeY) << "\n";
               out << SP << SP << "{\n";
               out << SP << SP << SP << fType << "* data = SOFIE::UTILITY::UnidirectionalBroadcast<" << fType << ">(tensor_" + fNInputs[i] << ", " << ConvertDimShapeToString(fShapeInputs[i]);
               out << ", " << ConvertDimShapeToString(fShapeY) << ");\n";
               out << SP << SP << SP << "std::copy(data, data + " << length << ", " << fNBroadcastedInputs[i] << ");\n";
               out << SP << SP << SP << "delete[] data;\n";
               out << SP << SP << "}\n";
            }
         }
      }

      if (fNInputs.size() == 1) {
         out << SP << "std::copy(tensor_" << fNInputs[0] << ", tensor_" << fNInputs[0] << " + ";
         out << length << ", tensor_" << fNY << ");\n";
      } else {
         std::vector<std::string> inputs(fNBroadcastedInputs.size());
         for (size_t i = 0; i < fNBroadcastedInputs.size(); i++) {
            inputs[i] = fNBroadcastedInputs[i] + "[id]";
         }
         out << SP << "for (size_t id = 0; id < " << length << "; id++) {\n";
         out << NaryOperatorTraits<T,Op>::Op("tensor_" + fNY + "[id]", inputs);
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string GetGPUCombine(const std::string& acc_v, const std::string& val) const {
      if (Op == EBasicNaryOperator::Max)
         return acc_v + " = (" + acc_v + " > " + val + ") ? " + acc_v + " : " + val + ";";
      if (Op == EBasicNaryOperator::Min)
         return acc_v + " = (" + acc_v + " < " + val + ") ? " + acc_v + " : " + val + ";";
      return acc_v + " = " + acc_v + " + " + val + ";"; // Sum and Mean both accumulate
   }

   // Per-input broadcast layout against the output: rank-padded shapes (dims), broadcast
   // masks (bcast), scalar/contiguous fast-path flags (isScalar/isContiguous) and the
   // dynamic shape params the kernel needs (dynParams). Filled by
   // GetGPUNaryBroadcastInfo() and used by both Generate_GPU_Kernel_ALPAKA (kernel
   // signature and index math) and Generate_GPU_ALPAKA (launch arguments), so the two
   // cannot drift apart.
   struct GPUNaryBroadcastInfo {
      std::vector<std::vector<Dim>> dims;
      std::vector<std::vector<bool>> bcast;
      std::vector<bool> isScalar;
      std::vector<bool> isContiguous;
      bool needCoords = false;
      std::vector<std::string> dynParams;
   };

   GPUNaryBroadcastInfo GetGPUNaryBroadcastInfo() const {
      GPUNaryBroadcastInfo info;
      const std::size_t nIn = fShapeInputs.size();
      const std::size_t D = fShapeY.size();
      info.dims.resize(nIn);
      info.bcast.resize(nIn);
      info.isScalar.assign(nIn, true);
      info.isContiguous.assign(nIn, true);
      bool anyGeneral = false;
      for (std::size_t i = 0; i < nIn; i++) {
         info.dims[i].assign(D, Dim{1});
         for (std::size_t k = 0; k < fShapeInputs[i].size(); k++)
            info.dims[i][D - fShapeInputs[i].size() + k] = fShapeInputs[i][k];
         info.bcast[i].resize(D);
         for (std::size_t d = 0; d < D; d++) {
            info.bcast[i][d] = !info.dims[i][d].isParam && info.dims[i][d].dim == 1;
            if (!info.bcast[i][d]) info.isScalar[i] = false;
            if (info.dims[i][d].GetVal() != fShapeY[d].GetVal()) info.isContiguous[i] = false;
         }
         if (!info.isScalar[i] && !info.isContiguous[i]) anyGeneral = true;
      }
      info.needCoords = anyGeneral;
      if (!info.needCoords)
         return info;

      UTILITY::CollectDimParams(UTILITY::ComputeStrideFromShape(fShapeY), info.dynParams);
      for (std::size_t i = 0; i < nIn; i++)
         if (!info.isScalar[i] && !info.isContiguous[i])
            UTILITY::CollectDimParams(UTILITY::ComputeStrideFromShape(info.dims[i]), info.dynParams);
      return info;
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      size_t nIn = fNInputs.size();
      auto info = GetGPUNaryBroadcastInfo();
      const std::size_t D = fShapeY.size();
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      std::string op;
      op += "\n//------ BASICNARY_KERNEL_ALPAKA\n";
      op += SP + "struct BasicNaryKernel_" + OpName + " {\n";
      op += SP + SP + "template<typename TAcc, typename TOut";
      for (size_t i = 0; i < nIn; i++)
         op += ", typename Tin" + std::to_string(i);
      op += ">\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc";
      for (size_t i = 0; i < nIn; i++)
         op += ", Tin" + std::to_string(i) + " const* in" + std::to_string(i);
      op += ", TOut* out";
      for (auto &p : info.dynParams)
         op += ", std::size_t const " + p;
      op += ", std::size_t n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";

      // decompose idx into output coords, skipping broadcast dims (stride 0)
      if (info.needCoords) {
         op += SP + SP + SP + "std::size_t remaining = idx;\n";
         op += SP + SP + SP + "std::size_t coord;\n";
         for (size_t i = 0; i < nIn; i++)
            if (!info.isScalar[i] && !info.isContiguous[i])
               op += SP + SP + SP + "std::size_t idx" + std::to_string(i) + " = 0;\n";
         for (std::size_t d = 0; d < D; d++) {
            std::string sY = "(" + stridesY[d].GetVal() + ")";
            op += SP + SP + SP + "coord = remaining / " + sY + ";\n";
            if (d + 1 < D)
               op += SP + SP + SP + "remaining -= coord * " + sY + ";\n";
            for (size_t i = 0; i < nIn; i++) {
               if (info.isScalar[i] || info.isContiguous[i] || info.bcast[i][d]) continue;
               auto strides = UTILITY::ComputeStrideFromShape(info.dims[i]);
               op += SP + SP + SP + "idx" + std::to_string(i) + " += coord * (" + strides[d].GetVal() + ");\n";
            }
         }
      }
      auto index = [&info](size_t i) -> std::string {
         if (info.isContiguous[i]) return "idx";
         if (info.isScalar[i]) return "0";
         return "idx" + std::to_string(i);
      };
      op += SP + SP + SP + "TOut v = static_cast<TOut>(in0[" + index(0) + "]);\n";
      for (size_t i = 1; i < nIn; i++)
         op += SP + SP + SP + "{ TOut w = static_cast<TOut>(in" + std::to_string(i) + "[" + index(i) + "]); " + GetGPUCombine("v", "w") + " }\n";
      if (Op == EBasicNaryOperator::Mean)
         op += SP + SP + SP + "v = v / static_cast<TOut>(" + std::to_string(nIn) + ");\n";
      op += SP + SP + SP + "out[idx] = v;\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string OpName) override {
      OpName = "op_" + OpName;
      return SP + "BasicNaryKernel_" + OpName + " basicNaryKernel_" + OpName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fShapeY.empty())
         throw std::runtime_error("SOFIE BasicNary Op called to Generate without being initialized first");
      OpName = "op_" + OpName;
      auto info = GetGPUNaryBroadcastInfo();
      std::stringstream out;
      std::string length = ConvertDimShapeToLength(fShapeY);
      out << "\n//------ BASICNARY_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerGrid_" << OpName << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "auto const workDiv_" << OpName << " = sofie_workdiv(elementsPerGrid_" << OpName << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << OpName
          << ", basicNaryKernel_" << OpName;
      for (auto &in : fNInputs)
         out << ", alpaka::getPtrNative(deviceBuf_" << in << ")";
      out << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")";
      for (auto &p : info.dynParams)
         out << ", static_cast<std::size_t>(" << p << ")";
      out << ", static_cast<std::size_t>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {return { std::string("cmath") }; }
};

}//SOFIE

#endif //SOFIE_ROPERATOR_BasicNary
