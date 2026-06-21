#ifndef SOFIE_ROPERATOR_TOPK
#define SOFIE_ROPERATOR_TOPK

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

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
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;
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

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() != 2) {
         throw std::runtime_error("SOFIE TopK Op Shape Inference needs exactly 2 input tensors");
      }

      auto shape = input[0]; // Shape format: [ m x n x o x p ... ]

      // set the dimension at the specified axis to k  (fAttrAxis is checked before that is in the correct range
      shape[fAttrAxis] = fK; // Modified shape: [ m x n x k x p ... ]
      return {shape, shape};
   }


   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNK) == false) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("SOFIE TopK Op Input Tensor i.e. K is not found in model");
      }

      fShapeX = model.GetTensorShape(fNX);
      auto fShapeK = model.GetTensorShape(fNK);
      auto kptr = static_cast<int64_t *>(model.GetInitializedTensorData(fNK).get());
      fK = *kptr;
      model.SetNotWritableInitializedTensor(fNK);
      fAttrAxis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      if(static_cast<size_t>(fAttrAxis) >=  fShapeX.size()){
         throw
            std::runtime_error("TMVA::SOFIE ONNX TopK op axis = "+ std::to_string(fAttrAxis) +" value exeeds size of tensor " +fNX+" of size "+std::to_string(fShapeX.size())+" .");
      }
      // fK cannot be larger that axis dimension
      fK = std::min(fK, fShapeX[fAttrAxis]);
      // if(fK>fShapeX[fAttrAxis]){
      //    throw
      //       std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(fK) +" value exeeds value of tensor " +fNX+" of size "+fShapeX.size()+" at axis= "+std::to_string(fAttrAxis)+".");
      // }
      // fShapeX = model.GetTensorShape(fNX); //  [ m x n x o x p ... ]
      // if(k[0]>=fShapeX.size()){
      //    throw
      //       std::runtime_error("TMVA::SOFIE ONNX TopK op k = "+ std::to_string(k[0]) +"value exeeds size of tensor " +fNX+" of size "+fShapeX.size()+" .");
      // }
      // fShapeY.push_back(2);
      // for (auto i : fShapeX)
      //    fShapeY.push_back(i); //  [ 2 x m x n x o x p ... ]
      // size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      // fShapeY[axis] = k[0]; //  [ 2 x m x n x K x p ... ]
      fShapeY=ShapeInference({fShapeX,fShapeK})[0];

      // for(int i=0;i<fShapeX.size();i++)
      // std::cout<<fShapeX[i]<<" ";
      // std::cout<<"\ny size -> "<<fShapeY.size()<<std::endl;


      model.AddIntermediateTensor(fNVal, model.GetTensorType(fNX), fShapeY);
      // output indices should be an int64 tensor
      model.AddIntermediateTensor(fNInd, ETensorType::INT64, fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
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

      size_t length=ConvertShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeX);
      // we perform loop on dimension before sorted axis and after sorted axis
      size_t n_before = (axis>0) ? length/strideX[axis-1] : 1;
      size_t n_after = strideX[axis];
      size_t n_elements = fShapeX[axis]; // number of elements to be sorted

      // }
      out << SP << "{\n"; // to define a separate scope for the operator code
      out << SP << "std::vector<std::pair<float,int64_t>> elements(" << n_elements << ");\n";
      // loop on elements before
      if (n_before > 1) {
         out << SP << "for (size_t i = 0; i < " << n_before << "; i++) {\n";
         out << SP << SP << "size_t xoffset = i*" << strideX[axis-1] << ";\n";
         out << SP << SP << "size_t yoffset = i*" << strideY[axis-1] << ";\n";
         out << SP;
      } else {
         out << SP << "size_t xoffset = 0;\n";
         out << SP << "size_t yoffset = 0;\n";
      }
      if (n_after > 1)
         out << SP << "for (size_t j = 0; j < " << n_after << "; j++) {\n";
      else
         out << SP << "const size_t j = 0;\n";

      // copy elements to be sorted in vector of pair
      out << SP << SP << "for (size_t l = 0; l < " << n_elements << "; l++) {\n";
      out << SP << SP << SP << "elements[l] = std::make_pair(tensor_" << fNX << "[xoffset + " << strideX[axis] << "*l + j], l);\n";
      out << SP << SP << "}\n";

      if (fAttrSorted) {
         if (fAttrLargest) {
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
               "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});\n";

         } else
            out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end()," <<
            "[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first<b.first) : a.second < b.second;});\n";
      } else
         // in this case we don;t need to return sorted elements, so we keep same order as before
         out<<SP<<SP << "std::partial_sort(elements.begin(),elements.begin()+" << fK << ",elements.end());\n";

      // copy the selected elements in the output
      out << SP << SP << "for (size_t l = 0; l < " << fK << "; l++) {\n";
      out << SP << SP << SP << "tensor_" << fNVal   << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].first;\n";
      out << SP << SP << SP << "tensor_" << fNInd << "[yoffset + " << strideY[axis] << "*l + j] = elements[l].second;\n";
      out << SP << SP << "}\n";
      if (n_after > 1) out << SP << SP << "}\n";
      if (n_before> 1) out << SP << "}\n";
      out << SP << "}\n"; // end operator scope
      return out.str();
   }

   // GPU baseline: one thread per slice along the sorted axis. Each thread keeps a
   // K-sized insertion-sorted buffer and selects its top-K in a single pass.
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis  = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      std::string NE  = std::to_string(fShapeX[axis]);
      std::string K   = std::to_string(fK);
      std::string CMP = fAttrLargest ? ">" : "<"; // best buffer ordered largest first pr smallest first
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
      op += SP + SP + SP + "std::size_t const strideXAxis,\n";
      op += SP + SP + SP + "std::size_t const strideXBefore,\n";
      op += SP + SP + SP + "std::size_t const strideYAxis,\n";
      op += SP + SP + SP + "std::size_t const strideYBefore) const {\n\n";

      op += SP + SP + SP + "auto const slice = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      // map the flat slice id to its (before, after) position and base offsets
      op += SP + SP + SP + "std::size_t const i = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const j = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = i * strideXBefore + j;\n";
      op += SP + SP + SP + "std::size_t const ybase = i * strideYBefore + j;\n\n";

      op += SP + SP + SP + "T bestV[" + K + "];\n";
      op += SP + SP + SP + "int64_t bestI[" + K + "];\n\n";

      // first K elements fill the buffer (K <= axis length is guaranteed in Initialize)
      op += SP + SP + SP + "for (int64_t l = 0; l < " + K + "; ++l) {\n";
      op += SP + SP + SP + SP + "T v = x[xbase + strideXAxis * (std::size_t)l];\n";
      op += SP + SP + SP + SP + "int64_t p = l;\n";
      op += SP + SP + SP + SP + "while (p > 0 && v " + CMP + " bestV[p-1]) { bestV[p] = bestV[p-1]; bestI[p] = bestI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + "bestV[p] = v; bestI[p] = l;\n";
      op += SP + SP + SP + "}\n\n";

      // remaining elements only compete with the current worst (bestV[K-1])
      op += SP + SP + SP + "for (int64_t l = " + K + "; l < " + NE + "; ++l) {\n";
      op += SP + SP + SP + SP + "T v = x[xbase + strideXAxis * (std::size_t)l];\n";
      op += SP + SP + SP + SP + "if (v " + CMP + " bestV[" + K + "-1]) {\n";
      op += SP + SP + SP + SP + SP + "int64_t p = " + K + "-1;\n";
      op += SP + SP + SP + SP + SP + "while (p > 0 && v " + CMP + " bestV[p-1]) { bestV[p] = bestV[p-1]; bestI[p] = bestI[p-1]; --p; }\n";
      op += SP + SP + SP + SP + SP + "bestV[p] = v; bestI[p] = l;\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      // buffer already ordered; ties keep the smaller index since l is scanned ascending
      op += SP + SP + SP + "for (int64_t s = 0; s < " + K + "; ++s) {\n";
      op += SP + SP + SP + SP + "vals[ybase + strideYAxis * (std::size_t)s] = bestV[s];\n";
      op += SP + SP + SP + SP + "inds[ybase + strideYAxis * (std::size_t)s] = bestI[s];\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";// end operator()
      op += SP + "};\n";// end struct
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "TopKKernel_" + fNVal + " topKernel_" + fNVal + ";\n";
   }

   // the geometry is computed here at codegen and passed as args matching the kernel signature.
   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      size_t length = ConvertShapeToLength(fShapeX);
      auto strideX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto strideY = UTILITY::ComputeStrideFromShape(fShapeY); // output is shorter along axis (K, not N_EL)
      size_t n_after = strideX[axis];
      size_t n_before = (axis > 0) ? length / strideX[axis-1] : 1;
      size_t numSlices = n_before * n_after;
      size_t strideX_axis = strideX[axis];
      size_t strideY_axis = strideY[axis];
      size_t strideX_before = (axis > 0) ? strideX[axis-1] : 0; // 0 is safe: i==0 when axis==0
      size_t strideY_before = (axis > 0) ? strideY[axis-1] : 0;

      std::stringstream out;
      out << "\n//-- TopK_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNVal << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNVal << " = Vec::all(Idx{" << numSlices << "});\n";
      out << SP << "auto const workDiv_" << fNVal << " = sofie_workdiv(elementsPerGrid_" << fNVal << ");\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNVal << ", topKernel_" << fNVal
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
          << ", static_cast<std::size_t>(" << numSlices     << "u)"
          << ", static_cast<std::size_t>(" << n_after       << "u)"
          << ", static_cast<std::size_t>(" << strideX_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideX_before<< "u)"
          << ", static_cast<std::size_t>(" << strideY_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideY_before<< "u));\n";
      return out.str();
   }

};

} // nameSPace SOFIE


#endif // SOFIE_ROPERATOR_TOPK
