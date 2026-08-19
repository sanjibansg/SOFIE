#ifndef SOFIE_ROPERATOR_CUMSUM
#define SOFIE_ROPERATOR_CUMSUM

#include "SOFIE/RModel.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_CumSum : public ROperator {
private:
   int fAxis = 0;
   int fExclusive;
   int fReverse;

   std::string fNX;
   std::string fNAxisTensor;
   std::string fNY;

   std::vector<Dim> fShape;
   std::string fType;
   size_t fRank;

public:
   ROperator_CumSum() {}

   ROperator_CumSum(const std::string &nameAxis, int exclusive, int reverse,
                    const std::string &nameX,
                    const std::string &nameY)
      : fExclusive(exclusive), fReverse(reverse),
        fNX(UTILITY::Clean_name(nameX)),
        fNAxisTensor(UTILITY::Clean_name(nameAxis)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::CUMSUM;
      fInputTensorNames  = { fNX, fNAxisTensor };
      fOutputTensorNames = { fNY };
   }

   ROperator_CumSum(int axis, int exclusive, int reverse,
                    const std::string &nameX,
                    const std::string &nameY)
      : fAxis(axis), fExclusive(exclusive), fReverse(reverse),
        fNX(UTILITY::Clean_name(nameX)),
        fNY(UTILITY::Clean_name(nameY))
   {
      fKind = OperatorKind::CUMSUM;
      fInputTensorNames  = { fNX };
      fOutputTensorNames = { fNY };
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   { return { input[0] }; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override
   { return { input[0] }; }

   void Initialize(RModel &model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("SOFIE CumSum: input tensor " + fNX + " not found");

      fShape = model.GetDimTensorShape(fNX);
      fRank  = fShape.size();
      fType  = ConvertTypeToString(model.GetTensorType(fNX));

      if (!fNAxisTensor.empty()) {
         if (!model.IsInitializedTensor(fNAxisTensor))
            throw std::runtime_error("SOFIE CumSum: axis tensor '" + fNAxisTensor +
                                     "' must be an initialized (constant) tensor");
         auto axisData = model.GetTensorData<int64_t>(fNAxisTensor);
         if (axisData.empty())
            throw std::runtime_error("SOFIE CumSum: axis tensor '" + fNAxisTensor + "' is empty");
         fAxis = static_cast<int>(axisData[0]);
      }

      // Resolve negative axis
      if (fAxis < 0) fAxis = static_cast<int>(fRank) + fAxis;
      if (fAxis < 0 || static_cast<size_t>(fAxis) >= fRank)
         throw std::runtime_error("SOFIE CumSum: axis " + std::to_string(fAxis) + " out of range for rank " + std::to_string(fRank));

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShape.empty())
         throw std::runtime_error("SOFIE CumSum " + opName + " called to Generate without being initialized");

      std::vector<std::string> dims(fRank);
      for (size_t i = 0; i < fRank; ++i) dims[i] = fShape[i].GetVal();
      auto strides = UTILITY::ComputeStrideFromShape(fShape);

      size_t ax = static_cast<size_t>(fAxis);
      std::string axDim = dims[ax];

      std::stringstream out;
      out << "\n//---- CumSum operator " << opName << " axis=" << fAxis
          << " exclusive=" << fExclusive << " reverse=" << fReverse << "\n";

      // Outer loops over all dims except the cumsum axis
      for (size_t i = 0; i < fRank; ++i) {
         if (i == ax) continue;
         out << SP << "for (size_t d_" << i << " = 0; d_" << i << " < " << dims[i] << "; ++d_" << i << ") {\n";
      }

      // running sum over the axis
      out << SP << SP << fType << " cs_acc = 0;\n";

      std::string ivar = "cs_i";
      if (!fReverse) {
         out << SP << SP << "for (size_t " << ivar << " = 0; " << ivar << " < " << axDim << "; ++" << ivar << ") {\n";
      } else {
         out << SP << SP << "for (size_t " << ivar << " = " << axDim << "; " << ivar << "-- > 0; ) {\n";
      }

      auto buildIdx = [&](const std::string &axVal) -> std::string {
         std::string idx = "";
         for (size_t i = 0; i < fRank; ++i) {
            if (!idx.empty()) idx += " + ";
            if (i == ax)
               idx += axVal + " * " + strides[i].GetVal();
            else
               idx += "d_" + std::to_string(i) + " * " + strides[i].GetVal();
         }
         return idx;
      };

      std::string inIdx  = buildIdx(ivar);
      std::string outIdx = buildIdx(ivar);

      if (fExclusive) {
         out << SP << SP << SP << "tensor_" << fNY << "[" << outIdx << "] = cs_acc;\n";
         out << SP << SP << SP << "cs_acc += tensor_" << fNX << "[" << inIdx << "];\n";
      } else {
         out << SP << SP << SP << "cs_acc += tensor_" << fNX << "[" << inIdx << "];\n";
         out << SP << SP << SP << "tensor_" << fNY << "[" << outIdx << "] = cs_acc;\n";
      }

      out << SP << SP << "}\n";

      // Close outer loops
      for (size_t i = 0; i < fRank; ++i) {
         if (i == ax) continue;
         out << SP << "}\n";
      }

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShape.empty())
         throw std::runtime_error("SOFIE CumSum GPU kernel called without initialization");

      size_t ax = static_cast<size_t>(fAxis);
      auto strides = UTILITY::ComputeStrideFromShape(fShape);

      std::vector<std::string> dims(fRank);
      for (size_t i = 0; i < fRank; ++i) dims[i] = fShape[i].GetVal();

      std::vector<Dim> outerShape;
      for (size_t i = 0; i < fRank; ++i)
         if (i != ax) outerShape.emplace_back(fShape[i]);
      std::string outerLen = ConvertDimShapeToLength(outerShape);
      auto outerStrides = UTILITY::ComputeStrideFromShape(outerShape);

      std::vector<size_t> outerDimIdx;
      for (size_t i = 0; i < fRank; ++i)
         if (i != ax) outerDimIdx.push_back(i);

      std::string kname = "CumSumKernel_" + opName;
      std::string exclStr = std::to_string(fExclusive);
      std::string revStr  = std::to_string(fReverse);

      std::string op;
      op  = "\n//------ CUMSUM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const outerLen,\n";
      op += SP + SP + SP + "std::size_t const axLen) const {\n\n";

      op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (tid >= outerLen) return;\n\n";

      if (!outerDimIdx.empty()) {
         for (size_t oi = 0; oi < outerDimIdx.size(); ++oi) {
            size_t di = outerDimIdx[oi];
            op += SP + SP + SP + "std::size_t const d_" + std::to_string(di)
               + " = (tid / " + outerStrides[oi].GetVal() + "u) % " + dims[di] + "u;\n";
         }
         op += "\n";
      }

      op += SP + SP + SP + "std::size_t const outer_base =";
      bool firstTerm = true;
      for (size_t di : outerDimIdx) {
         if (!firstTerm) op += " +";
         op += " d_" + std::to_string(di) + " * " + strides[di].GetVal() + "u";
         firstTerm = false;
      }
      if (firstTerm) op += " 0u";
      op += ";\n\n";

      // axis stride
      op += SP + SP + SP + "std::size_t const ax_stride = " + strides[ax].GetVal() + "u;\n\n";

      // cumsum loop
      std::string excl = (fExclusive ? "true" : "false");
      if (!fReverse) {
         op += SP + SP + SP + "T acc_val = static_cast<T>(0);\n";
         op += SP + SP + SP + "for (std::size_t i = 0; i < axLen; ++i) {\n";
         op += SP + SP + SP + SP + "std::size_t idx = outer_base + i * ax_stride;\n";
         if (fExclusive) {
            op += SP + SP + SP + SP + "Y[idx] = acc_val;\n";
            op += SP + SP + SP + SP + "acc_val += X[idx];\n";
         } else {
            op += SP + SP + SP + SP + "acc_val += X[idx];\n";
            op += SP + SP + SP + SP + "Y[idx] = acc_val;\n";
         }
         op += SP + SP + SP + "}\n";
      } else {
         op += SP + SP + SP + "T acc_val = static_cast<T>(0);\n";
         op += SP + SP + SP + "for (std::size_t i = axLen; i-- > 0; ) {\n";
         op += SP + SP + SP + SP + "std::size_t idx = outer_base + i * ax_stride;\n";
         if (fExclusive) {
            op += SP + SP + SP + SP + "Y[idx] = acc_val;\n";
            op += SP + SP + SP + SP + "acc_val += X[idx];\n";
         } else {
            op += SP + SP + SP + SP + "acc_val += X[idx];\n";
            op += SP + SP + SP + SP + "Y[idx] = acc_val;\n";
         }
         op += SP + SP + SP + "}\n";
      }

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "CumSumKernel_" + opName;
      return SP + kname + " cumSumKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShape.empty())
         throw std::runtime_error("SOFIE CumSum GPU dispatch called without initialization");

      size_t ax = static_cast<size_t>(fAxis);
      std::vector<Dim> outerShape;
      for (size_t i = 0; i < fRank; ++i)
         if (i != ax) outerShape.emplace_back(fShape[i]);
      std::string outerLen = ConvertDimShapeToLength(outerShape);
      std::string axLen    = fShape[ax].GetVal();

      std::stringstream out;
      out << "\n//------ CUMSUM_GPU_ALPAKA\n";
      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << outerLen << "});\n";
      out << SP << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", cumSumKernel_" << opName << ", "
          << "alpaka::getPtrNative(deviceBuf_" << fNX << "), "
          << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
          << "static_cast<Idx>(" << outerLen << "), "
          << "static_cast<Idx>(" << axLen << "));\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_CUMSUM
