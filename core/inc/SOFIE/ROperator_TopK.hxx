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

   // next power of two >= the axis length (the bitonic network needs a power-of-2 size)
   size_t TopKPaddedAxis() const {
      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      size_t n = fShapeX[axis];
      size_t p = 1; while (p < n) p <<= 1;
      return p;
   }
   // threads per block: cover the paddedN/2 comparator pairs, capped at 1024, one warp
   // kernel and launch both call this
   size_t TopKBlockThreads() const {
      size_t pairs = TopKPaddedAxis() / 2;
      size_t bt = (pairs < 1024) ? pairs : 1024;
      if (bt < 32) bt = 32;
      return bt;
   }

   // We have one block per slice. Cache the row in shared memory, bitonic-sort it
   // best-first (indices ride along for the tie-break), then write the first K.
   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override {
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE Operator TopK called to Generate without being initialized first");

      size_t axis = fAttrAxis < 0 ? fShapeX.size() + fAttrAxis : fAttrAxis;
      std::string NE = std::to_string(fShapeX[axis]); // real axis length
      std::string PAD = std::to_string(TopKPaddedAxis()); // next power of two >= NE
      std::string PADH = std::to_string(TopKPaddedAxis() / 2); // number of comparator pairs
      std::string BT = std::to_string(TopKBlockThreads()); // threads per block
      std::string K = std::to_string(fK);
      std::string OP = fAttrLargest ? ">" : "<"; // best-first value comparator
      std::string SENT = fAttrLargest ? "std::numeric_limits<T>::lowest()"
                                      : "std::numeric_limits<T>::max()"; // padded slots never win
      std::string kname = "TopKKernel_" + fNVal;

      // shared-memory budget guard (caches the whole padded row)
      size_t valBytes = (fType == "double" || fType == "int64_t") ? 8 : 4;
      if (TopKPaddedAxis() * (valBytes + 8) > 48u * 1024u)
         throw std::runtime_error("SOFIE TopK GPU: axis length " + NE +
            " too long for shared-memory bitonic top-K");

      std::string op;
      op  = "\n//------ TopK_KERNEL_ALPAKA (block-per-row bitonic)\n";
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

      // shared row buffers (values + indices), declared before the early return
      op += SP + SP + SP + "auto& sv = alpaka::declareSharedVar<T[" + PAD + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto& si = alpaka::declareSharedVar<int64_t[" + PAD + "], __COUNTER__>(acc);\n";
      op += SP + SP + SP + "auto const slice = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
      op += SP + SP + SP + "auto const tid   = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (slice >= numSlices) return;\n\n";

      op += SP + SP + SP + "std::size_t const ib = slice / nAfter;\n";
      op += SP + SP + SP + "std::size_t const jb = slice % nAfter;\n";
      op += SP + SP + SP + "std::size_t const xbase = ib * strideXBefore + jb;\n";
      op += SP + SP + SP + "std::size_t const ybase = ib * strideYBefore + jb;\n\n";

      // 1) load row into shared, pad the tail with a sentinel that never wins
      op += SP + SP + SP + "for (std::size_t l = tid; l < " + PAD + "u; l += " + BT + "u) {\n";
      op += SP + SP + SP + SP + "if (l < " + NE + "u) { sv[l] = x[xbase + strideXAxis * l]; si[l] = (int64_t)l; }\n";
      op += SP + SP + SP + SP + "else { sv[l] = " + SENT + "; si[l] = (int64_t)" + NE + "; }\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

      // 2) bitonic sort: kk = bitonic seq size, jj = compare distance, best ends at index 0.
      // each thread owns ONE comparator pair (i, i^jj)
      op += SP + SP + SP + "for (std::size_t kk = 2u; kk <= " + PAD + "u; kk <<= 1) {\n";
      op += SP + SP + SP + SP + "for (std::size_t jj = kk >> 1; jj > 0u; jj >>= 1) {\n";
      op += SP + SP + SP + SP + SP + "for (std::size_t t = tid; t < " + PADH + "u; t += " + BT + "u) {\n";
      op += SP + SP + SP + SP + SP + SP + "std::size_t const i = ((t & ~(jj - 1u)) << 1) | (t & (jj - 1u));\n";
      op += SP + SP + SP + SP + SP + SP + "std::size_t const p = i | jj;\n";
      op += SP + SP + SP + SP + SP + SP + "T av = sv[i]; T bv = sv[p];\n";
      op += SP + SP + SP + SP + SP + SP + "int64_t ai = si[i]; int64_t bi = si[p];\n";
      op += SP + SP + SP + SP + SP + SP + "bool const firstFirst = (av " + OP + " bv) || (av == bv && ai < bi);\n";
      op += SP + SP + SP + SP + SP + SP + "bool const dir = ((i & kk) == 0u);\n";
      op += SP + SP + SP + SP + SP + SP + "bool const sw  = (firstFirst != dir);\n";
      op += SP + SP + SP + SP + SP + SP + "sv[i] = sw ? bv : av; sv[p] = sw ? av : bv;\n";
      op += SP + SP + SP + SP + SP + SP + "si[i] = sw ? bi : ai; si[p] = sw ? ai : bi;\n";
      op += SP + SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n\n";

      // 3) write top-K (already best-first)
      op += SP + SP + SP + "for (std::size_t s = tid; s < " + K + "u; s += " + BT + "u) {\n";
      op += SP + SP + SP + SP + "vals[ybase + strideYAxis * s] = sv[s];\n";
      op += SP + SP + SP + SP + "inds[ybase + strideYAxis * s] = si[s];\n";
      op += SP + SP + SP + "}\n";

      op += SP + SP + "}\n";// end operator()
      op += SP + "};\n";// end struct
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override {
      return SP + "TopKKernel_" + fNVal + " topKernel_" + fNVal + ";\n";
   }

   std::vector<std::string> GetStdLibs() override {
      return { std::string("limits") };
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
      const size_t blockThreads = TopKBlockThreads();
      out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << fNVal << "(\n";
      out << SP << SP << "Vec::all(static_cast<Idx>(" << numSlices << ")),\n";
      out << SP << SP << "Vec::all(Idx{" << blockThreads << "u}),\n";
      out << SP << SP << "Vec::all(Idx{1u}));\n";
      out << SP << "auto task_" << fNVal << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNVal << ", topKernel_" << fNVal
          << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNVal << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNInd << ")"
          << ", static_cast<std::size_t>(" << numSlices     << "u)"
          << ", static_cast<std::size_t>(" << n_after       << "u)"
          << ", static_cast<std::size_t>(" << strideX_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideX_before<< "u)"
          << ", static_cast<std::size_t>(" << strideY_axis  << "u)"
          << ", static_cast<std::size_t>(" << strideY_before<< "u));\n";
      out << SP << "alpaka::enqueue(queue, task_" << fNVal << ");\n";
      return out.str();
   }

};

} // nameSPace SOFIE


#endif // SOFIE_ROPERATOR_TOPK
