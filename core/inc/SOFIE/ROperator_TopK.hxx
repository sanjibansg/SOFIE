#ifndef SOFIE_ROPERATOR_TOPK
#define SOFIE_ROPERATOR_TOPK

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <algorithm>
#include <sstream>


namespace SOFIE {

template <typename T>
class ROperator_TopK final : public ROperator {

private:
   int fAttrAxis;
   int fAttrLargest;
   int fAttrSorted;

   size_t fK;
   std::string fNK;
   std::string fNX;
   std::string fNVal;
   std::string fNInd;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;
   std::string fType;

public:
   ROperator_TopK() {}
   ROperator_TopK(int attr_axis, int attr_largest, int attr_sorted, std::string nameK, std::string nameX, std::string nameVal, std::string nameInd)
      : fAttrAxis(attr_axis),
        fAttrLargest(attr_largest),
        fAttrSorted(attr_sorted),
        fNK(UTILITY::Clean_name(nameK)),
        fNX(UTILITY::Clean_name(nameX)),
        fNVal(UTILITY::Clean_name(nameVal)),
        fNInd(UTILITY::Clean_name(nameInd)){
            fInputTensorNames = { fNX, fNK };
            fOutputTensorNames = { fNVal, fNInd };
        }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
         ETensorType ret = input[0];
         return {ret, ret};
      }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         throw std::runtime_error("SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         throw std::runtime_error("SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);
      auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      size_t kval = *kptr;
      model.SetNotWritableInitializedTensor(fNK);
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if (static_cast<size_t>(fAttrAxis) >= fShapeX.size()) {
         throw std::runtime_error("TMVA::SOFIE ONNX TopK op axis = " + std::to_string(fAttrAxis) +
            " value exceeds size of tensor " + fNX + " of size " + std::to_string(fShapeX.size()) + " .");
      }
      fK = fShapeX[fAttrAxis].isParam ? kval : std::min(kval, fShapeX[fAttrAxis].dim);

      fShapeY = fShapeX;
      fShapeY[fAttrAxis] = Dim{fK};

      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));

      if (model.Verbose()) {
         std::cout << "TopK " << fNX << "  " << ConvertDimShapeToString(fShapeX)
                   << "---> " << fNVal << " " << ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t size = fShapeX.size();
      size_t axis = fAttrAxis < 0 ? size + fAttrAxis : fAttrAxis;
      out << "\n" << SP << "//------ TopK\n";

      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      std::vector<Dim> shape_before(fShapeX.begin(), fShapeX.begin() + axis);
      std::string n_before = (axis > 0) ? ConvertDimShapeToLength(shape_before) : "1";
      std::string n_after = strideX[axis].GetVal();
      std::string n_elements = fShapeX[axis].GetVal();

      out << SP << "{\n";
      out << SP << "std::vector<std::pair<float,int64_t>> elements(" << n_elements << ");\n";
      if (n_before != "1") {
         out << SP << "for (size_t i = 0; i < " << n_before << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << strideX[axis-1] << ";\n";
         out << SP << SP << "size_t yoffset = i*" << strideY[axis-1] << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (n_after != "1")
         out << SP << "for (size_t j = 0; j < " << n_after << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      out << SP << SP << "for (size_t l = 0; l < " << n_elements << "; l++) {\n";
      out << SP << SP << SP << "elements[l] = std::make_pair(tensor_" << fNX << "[xoffset + " << strideX[axis] << "*l + j], l);\n";
      out << SP << SP << "}\n";

      if (fAttrSorted) {
         if (fAttrLargest)
            out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end(),"
                << "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});\n";
         else
            out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end(),"
                << "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first<b.first) : a.second < b.second;});\n";
      } else
         out << SP << SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end());\n";

      out << SP << SP << "for (size_t l = 0; l < " << fK << "; l++) {\n";
      out << SP << SP << SP << "tensor_" << fNVal << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].first;\n";
      out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].second;\n";
      out << SP << SP << "}\n";
      if (n_after != "1") out << SP << SP << "}\n";
      if (n_before != "1") out << SP << "}\n";
      out << SP << "}\n";
      return out.str();
   }

   // one thread per slice with a K-sized insertion-sorted register buffer; the axis
   // length is a runtime arg so the kernel handles a dynamic axis
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      std::string K   = std::to_string(fK);
      std::string CMP = fAttrLargest ? ">" : "<";
      std::string kname = "TopKKernel_" + fNVal;

      std::string op;
      op  = "\n//------ TopK_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ x,\n";
      op += SP + SP + SP + "T* __restrict__ vals,\n";
      op += SP + SP + SP + "int64_t* __restrict__ inds,\n";
      op += SP + SP + SP + "std::size_t const numSlices,\n";
      op += SP + SP + SP + "std::size_t const nAfter,\n";
      op += SP + SP + SP + "std::size_t const nElAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXBefore,\n";
      op += SP + SP + SP + "std::size_t const strideYAxis,\n";
      op += SP + SP + SP + "std::size_t const strideYBefore) const {\n\n";

      op += SP + SP + SP + "auto const slice = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      op += SP + SP + SP + "std::size_t const i = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const j = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = i * strideXBefore + j;\n";
      op += SP + SP + SP + "std::size_t const ybase = i * strideYBefore + j;\n\n";

      op += SP + SP + SP + "T bestV[" + K + "];\n";
      op += SP + SP + SP + "int64_t bestI[" + K + "];\n\n";

      op += SP + SP + SP + "for (int64_t l = 0; l < " + K + "; ++l) {\n";
      op += SP + SP + SP + SP + "T v = x[xbase + strideXAxis * (std::size_t)l];\n";
      op += SP + SP + SP + SP + "int64_t p = l;\n";
      op += SP + SP + SP + SP + "while (p > 0 && v " + CMP + " bestV[p-1]) { bestV[p] = bestV[p-1]; bestI[p] = bestI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + "bestV[p] = v; bestI[p] = l;\n";
      op += SP + SP + SP + "}\n\n";

      op += SP + SP + SP + "for (int64_t l = " + K + "; l < (int64_t)nElAxis; ++l) {\n";
      op += SP + SP + SP + SP + "T v = x[xbase + strideXAxis * (std::size_t)l];\n";
      op += SP + SP + SP + SP + "if (v " + CMP + " bestV[" + K + "-1]) {\n";
      op += SP + SP + SP + SP + SP + "int64_t p = " + K + "-1;\n";
      op += SP + SP + SP + SP + SP + "while (p > 0 && v " + CMP + " bestV[p-1]) { bestV[p] = bestV[p-1]; bestI[p] = bestI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + SP + "bestV[p] = v; bestI[p] = l;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      op += SP + SP + SP + "for (int64_t s = 0; s < " + K + "; ++s) {\n";
      op += SP + SP + SP + SP + "vals[ybase + strideYAxis * (std::size_t)s] = bestV[s];\n";
      op += SP + SP + SP + SP + "inds[ybase + strideYAxis * (std::size_t)s] = bestI[s];\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "TopKKernel_" + fNVal + " topKernel_" + fNVal + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY);
      std::vector<Dim> shape_before(fShapeX.begin(), fShapeX.begin() + axis);
      std::string n_before = (axis > 0) ? ConvertDimShapeToLength(shape_before) : "1";
      std::string n_after = strideX[axis].GetVal();
      std::string numSlices = "((" + n_before + ")*(" + n_after + "))";
      std::string nElAxis = fShapeX[axis].GetVal();
      std::string strideX_axis = strideX[axis].GetVal();
      std::string strideY_axis = strideY[axis].GetVal();
      std::string strideX_before = (axis > 0) ? strideX[axis-1].GetVal() : "0";
      std::string strideY_before = (axis > 0) ? strideY[axis-1].GetVal() : "0";

      std::stringstream out;
      out << "\n//-- TopK_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNVal << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNVal << " = Vec::all(static_cast<Idx>(" << numSlices << "));\n";
      out << SP << "auto const workDiv_" << fNVal << " = sofie_workdiv(elementsPerGrid_" << fNVal << ");\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNVal << ", topKernel_" << fNVal
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
          << ", static_cast<std::size_t>(" << numSlices      << ")"
          << ", static_cast<std::size_t>(" << n_after        << ")"
          << ", static_cast<std::size_t>(" << nElAxis        << ")"
          << ", static_cast<std::size_t>(" << strideX_axis   << ")"
          << ", static_cast<std::size_t>(" << strideX_before << ")"
          << ", static_cast<std::size_t>(" << strideY_axis   << ")"
          << ", static_cast<std::size_t>(" << strideY_before << "));\n";
      return out.str();
   }

};

} // namespace SOFIE


#endif // SOFIE_ROPERATOR_TOPK
