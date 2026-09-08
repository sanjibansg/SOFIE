#ifndef SOFIE_ROPERATOR_LAYERNORMALIZATION
#define SOFIE_ROPERATOR_LAYERNORMALIZATION

#include "SOFIE/RModel.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/SOFIE_common.hxx"
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

namespace SOFIE {

template <typename T>
class ROperator_LayerNormalization : public ROperator {
private:
   bool fCastToFloat = false;  // flag to indicate if operation 1 are in floats (to be  impl)
   int fAttrAxis;
   float fAttrEpsilon;
   size_t fAttrStashType;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNY;
   std::string fNMean;
   std::string fNInvStdDev;

   std::string fNCastedX;
   std::string fNNormalizedX;
   std::string fNBroadcastedB;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeScale;
   std::vector<Dim> fShapeB;
   std::vector<Dim> fShapeY;
   std::vector<Dim> fShapeMean;
   std::vector<Dim> fShapeInvStdDev;

   size_t fAxis; // axis in [0, size)
   size_t fSize; // Size of the input
   // size_t fAxisDim;

   std::vector<Dim> fNormalizedShape;  // shape from X[ axis,...,N-1]
   std::vector<Dim> fAxesShape;        // shape from X[0,..,axis-1]
   // lengths in string format
   std::string fLength; // Length of the input
   std::string fNormalizedLength;
   std::string fAxesLength;

   std::string fType;

   // Fold grid: the store applies the absorbed boundary's snap and keeps writing float.
   // Carrier grid: the store encodes and writes the boundary's code tensor instead.
   std::optional<QuantizationGrid> fFakeQuantGrid;
   std::optional<QuantizationGrid> fCarrierGrid;

public:
   ROperator_LayerNormalization() {}

   ROperator_LayerNormalization(int axis, float epsilon, size_t stashType, const std::string &nameX,
                                const std::string &nameScale, const std::string &nameB, const std::string &nameY,
                                const std::string &nameMean, const std::string &nameInvStdDev)
      : fAttrAxis(axis), fAttrEpsilon(epsilon), fAttrStashType(stashType), fNX(UTILITY::Clean_name(nameX)),
        fNScale(UTILITY::Clean_name(nameScale)), fNB(UTILITY::Clean_name(nameB)),
        fNY(UTILITY::Clean_name(nameY)), fNMean(UTILITY::Clean_name(nameMean)), fNInvStdDev(UTILITY::Clean_name(nameInvStdDev))
   {
         fInputTensorNames = { fNX, fNScale };
         if (!fNB.empty()){
            fInputTensorNames.emplace_back(fNB);
         }

         fOutputTensorNames = { fNY };
         if (!fNMean.empty()){
            fOutputTensorNames.emplace_back(fNMean);
         }
         if (!fNInvStdDev.empty()){
            fOutputTensorNames.emplace_back(fNInvStdDev);
         }
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Tensor " + fNX + " not found.");
      }
      bool isDynamic = model.IsDynamicTensor(fNX);
      fShapeX = model.GetDimTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      // Type of the output
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      // Size of the input
      fSize = fShapeX.size();
      // Axis in [0, size)
      fAxis = (fAttrAxis < 0) ? fSize + fAttrAxis : fAttrAxis;
      // Shape of fShapeX[0, ..., fAxis)
      fAxesShape = std::vector<Dim>(fShapeX.begin(), fShapeX.begin() + fAxis);
      // Length of the axes
      fAxesLength = ConvertDimShapeToLength(fAxesShape);
      // Shape of fShapeX[fAxis, ..., fSize)
      fNormalizedShape = std::vector<Dim>(fShapeX.begin() + fAxis, fShapeX.end());
      // Length of the normalized axis
      fNormalizedLength = ConvertDimShapeToLength(fNormalizedShape);
      // length of the input
      fLength = ConvertDimShapeToLength(fShapeX);
      // Type of mean and std
      ETensorType type = (fAttrStashType == 1) ? ETensorType::FLOAT : model.GetTensorType(fNX);
      // Mean
      if (!fNMean.empty()) {
         // cannot use initializer list with one element since it is ambiguous
         if (isDynamic)
            // add size_t(-1) to indicate that shape is an expression
            model.AddIntermediateTensor(fNMean, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNMean, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // Inverse Standard Deviation
      if (!fNInvStdDev.empty()) {
         if (isDynamic)
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<Dim>(1,Dim{fAxesLength,std::size_t(-1)}));
         else
            model.AddIntermediateTensor(fNInvStdDev, type, std::vector<size_t>(1,std::stoi(fAxesLength)));
      }
      // if mean and stdev are not empty they are not defined in the output list
      // Cast X to float
      if (fAttrStashType == 1 && model.GetTensorType(fNX) != ETensorType::FLOAT) {
         fCastToFloat = true;
         fType = "float";
         // fNCastedX = "Casted" + fNX;
         // model.AddIntermediateTensor(fNCastedX, ETensorType::FLOAT, fShapeX);
         // fNNormalizedX = "Normalized" + fNX;
         // model.AddIntermediateTensor(fNNormalizedX, ETensorType::FLOAT, fShapeX);
      }
      // scale shape
      fShapeScale = model.GetDimTensorShape(fNScale);
      // appends 1 to scale shapes if missing
      size_t dimScale = fShapeScale.size();
      if (dimScale < fSize) {
         for (size_t i = 0; i < fSize-dimScale; i++)
            fShapeScale.insert(fShapeScale.begin(), Dim{1});
      }
      // check also shape if consistent now
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeScale[i].dim != 1 && fShapeScale[i] != fShapeX[i])
            throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Scale Tensor has invalid shape " + ConvertDimShapeToString(fShapeScale));
      }
      if (!fNB.empty()) {
         fShapeB = model.GetDimTensorShape(fNB);
         // appends 1 to bias shapes if missing
         size_t dimB = fShapeB.size();
         if (dimB < fShapeX.size()) {
            for (size_t i = 0; i < fSize-dimB; i++)
               fShapeB.insert(fShapeB.begin(), Dim{1});
         }
         for (size_t i = 0; i < fSize; i++) {
            if (fShapeB[i].dim != 1 && fShapeB[i] != fShapeX[i])
               throw std::runtime_error("TMVA::SOFIE - LayerNormalization - Bias Tensor has invalid shape " + ConvertDimShapeToString(fShapeScale));
         }
      }

      std::cout << "bias + scale " << ConvertDimShapeToString(fShapeB) << "  " << ConvertDimShapeToString(fShapeScale) << std::endl;

      // // Broadcast the bias
      // if (!fNB.empty()) {
      //    fShapeB = model.GetTensorShape(fNB);
      //    size_t lengthB = ConvertShapeToLength(fShapeB);
      //    if (isDynamic || lengthB < static_cast<size_t>(std::stoi(fLength))) {
      //       fNBroadcastedB = "Broadcasted" + fNB;
      //       model.AddIntermediateTensor(fNBroadcastedB, ConvertStringToType(fType), fShapeX);
      //    }
      // }
      model.AddNeededStdLib("cmath");
   }

   // Absorbs a downstream boundary in either mode: the snap runs before the store, or the
   // store encodes onto `grid` and writes the boundary's code tensor.
   bool CanFuseOutputOnGrid(EQuantizedOutputEmit mode) const override
   {
      return mode == EQuantizedOutputEmit::Snap || (!fFakeQuantGrid && !fCarrierGrid);
   }

   // Adopts the boundary's output tensor; the previous output is left for dead-code
   // elimination.
   void FuseOutputOnGrid(const std::string &output, const QuantizationGrid &grid,
                         EQuantizedOutputEmit mode) override
   {
      AdoptFusedOutputName(fNY, fOutputTensorNames, output, "LayerNormalization");
      if (mode == EQuantizedOutputEmit::Snap)
         fFakeQuantGrid = grid;
      else
         fCarrierGrid = grid;
   }

   std::string GenerateInitCode() override
   {
      std::stringstream out;
      if (!fNBroadcastedB.empty()) {
         out << SP << "// Broadcasting the bias of LayerNormalization op\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = SOFIE::UTILITY::UnidirectionalBroadcast(tensor_";
         out << fNB << ", " << ConvertDimShapeToString(fShapeB) << ", " << ConvertDimShapeToString(fShapeX) << ");\n";
         out << SP << "std::copy(data, data + " << fLength << ", tensor_" << fNBroadcastedB << ");\n";
         out << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate(std::string opName) override
   {
      opName = "op_" + opName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator " + opName +
                                  " called to generate without being initialized first.");
      }

      std::stringstream out;

      out << "//---- Layer Normalization  operator " << opName << "\n";

      // Loop over all the normalized axes i.e. [axis, ..., size)
      std::vector<std::string> inputShape(fSize);

      for (size_t i = 0; i < fSize; i++) {
         inputShape[i] = fShapeX[i].GetVal();
      }

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string inputIndex = "axis_0 * " + strides[0].GetVal();
      for (size_t i = 1; i < fSize; i++) {
         inputIndex += " + axis_" + std::to_string(i);
         if (i < fSize-1) inputIndex += " * " + strides[i].GetVal();
      }
      auto scaleStrides = UTILITY::ComputeStrideFromShape(fShapeScale);
      std::string scaleIndex;
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeScale[i].dim != 1) {
            if (!scaleIndex.empty()) scaleIndex += " + ";
            scaleIndex += "axis_" + std::to_string(i);
            if ( scaleStrides[i].dim != 1) scaleIndex +=  " * " + scaleStrides[i].GetVal();
         }
      }
      if (scaleIndex.empty()) scaleIndex = "0";

      auto biasStrides = UTILITY::ComputeStrideFromShape(fShapeB);
      std::string biasIndex;
      for (size_t i = 0; i < fSize; i++) {
         if (fShapeB[i].dim != 1) {
            if (!biasIndex.empty()) biasIndex += " + ";
            biasIndex += "axis_" + std::to_string(i);
            if ( biasStrides[i].dim != 1) biasIndex +=  " * " + biasStrides[i].GetVal();
         }
      }
      if (biasIndex.empty()) biasIndex = "0";

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex = "axis_" + std::to_string(0) + " * " + axesStrides[0].GetVal();
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + axesStrides[i].GetVal();
      }


      // compute mean and std-dev. Save in tensors if requested

      out << SP << "// Compute the mean\n";

      // Loop over all the outer dims in [0, fAxis)
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape[i]
                      << "; " << iIdx << "++) {\n";
      }
      out << SP << SP << fType << " mean = 0.;\n";
      // loop over the normalized dimensions (fAxis,....,N-1)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                         << "; " << jIdx << "++) {\n";
      }
      out << SP << SP << SP << "mean += tensor_" << fNX << "[" << inputIndex << "];\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "mean  /= " << fType << "(" << fNormalizedLength << ");\n";


      out << SP << "// Compute the inverse Standard Deviation\n";

      // Set sum = 0
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j]
                          << "; " << jIdx << "++){\n";
      }
      out << SP << SP << SP << "float tmp = tensor_" << fNX << "[" << inputIndex << "] - mean;\n";
      out << SP << SP << SP << "sum += tmp*tmp;\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << fType << " invStdDev = 1 / std::sqrt(";
      out << "sum / " << fType << "(" << fNormalizedLength << ") + " << fAttrEpsilon << ");\n";


      // set output mean and invStdDev if requested
      if (!fNMean.empty())
         out << SP << SP <<  "tensor_" << fNMean << "[" << axesIndex << "] = mean;\n";
      if (!fNInvStdDev.empty())
         out << SP << SP <<  "tensor_" << fNInvStdDev << "[" << axesIndex << "] = invStdDev;\n";

      // scale and add bias

      out << SP << "// Y = Scale o InvStdDev (X - Mean)\n";

      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape[j] << "; " << jIdx
             << "++){\n";
      }
      out << SP << SP << SP << "tensor_" << fNY << "[" << inputIndex << "] = tensor_" << fNScale;
      out << "[" << scaleIndex << "] * invStdDev * (tensor_" << fNX << "[" << inputIndex << "] - mean)";

      // add bias if needed
      if (!fNB.empty())
         // assume bias has index as scale
         out << " + tensor_" << fNB << "[" << biasIndex << "]";
      out << ";\n";

      // close loops on normalizing dim  [..,fAxis,...fSize-1]
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      // close loops on the other dimensions [0,...,fAxis]
      for (size_t i = 0; i < fAxis; i++) {
         out << SP << "}\n";
      }

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("TMVA::SOFIE LayerNormalization called to Generate without being initialized first");

      // Static normalizedLength <= 1024 runs one block per row with shared-memory tree
      // reductions; dynamic shapes or longer rows use the serial one-thread-per-row kernel.
      size_t normLenVal = 0;
      bool canParallel = false;
      try {
         normLenVal = std::stoul(fNormalizedLength);
         canParallel = (normLenVal > 0 && normLenVal <= 1024);
      } catch (...) {}

      // Compute blockSize = next power-of-2 >= normLenVal
      size_t blockSize = 1;
      if (canParallel) {
         while (blockSize < normLenVal) blockSize <<= 1;
      }

      // axesLength = product of fShapeX[0..axis), normalizedLength = product of
      // fShapeX[axis..size); a row is one axes element spanning the normalized dims.

      std::vector<std::string> inputShape(fSize);
      for (size_t i = 0; i < fSize; i++)
         inputShape[i] = fShapeX[i].GetVal();

      auto strides      = UTILITY::ComputeStrideFromShape(fShapeX);
      auto scaleStrides = UTILITY::ComputeStrideFromShape(fShapeScale);
      auto biasStrides  = (!fNB.empty()) ? UTILITY::ComputeStrideFromShape(fShapeB)
                                          : std::vector<Dim>{};
      auto axesStrides  = UTILITY::ComputeStrideFromShape(fAxesShape);

      // The kernel decomposes the linear thread index into axis coordinates and loops over
      // the normalized dims, mirroring Generate()'s index expressions.

      std::string kname = "LayerNormKernel_" + opName;
      std::string op;
      op  = "\n//------ LAYERNORM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T, typename TOut>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T const* __restrict__ scale,\n";
      if (!fNB.empty())
         op += SP + SP + SP + "T const* __restrict__ bias,\n";
      if (!fNMean.empty())
         op += SP + SP + SP + "T* __restrict__ out_mean,\n";
      if (!fNInvStdDev.empty())
         op += SP + SP + SP + "T* __restrict__ out_invstd,\n";
      op += SP + SP + SP + "TOut* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const axesLength) const {\n\n";

      if (canParallel) {
         // Parallel path: one block per row, one thread per normalized element; two
         // shared-memory tree reductions compute mean then variance.
         std::string bs = std::to_string(blockSize);
         std::string nl = fNormalizedLength; // e.g. "64"
         std::string eps = std::to_string(fAttrEpsilon);

         op += SP + SP + SP + "// Block-parallel LayerNorm: one block per row, "
               + bs + " threads per block, " + nl + " active.\n";
         op += SP + SP + SP + "auto& shmem = alpaka::declareSharedVar<T[" + bs + "], __COUNTER__>(acc);\n";
         op += SP + SP + SP + "auto const row = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];\n";
         op += SP + SP + SP + "auto const tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (row >= axesLength) return;\n\n";

         // --- Decompose row into axis-dim coordinates (same logic as serial path) ---
         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + axesStrides[i].GetVal() + "u) % "
                  + inputShape[i] + "u;\n";
            }
            op += "\n";
         }

         // Base input offset for this row
         op += SP + SP + SP + "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += SP + SP + SP + SP + "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + "axis_" + std::to_string(i)
                  + " * " + strides[i].GetVal() + "u";
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         // Maps thread id to a normalized-dims offset via the code-gen-time inner strides:
         // norm_offset = sum_j (tid / normInnerStride[j]) % dim[j] * stride[j].

         // Build the norm-dim strides (strides within the flattened normalised space)
         auto normShape = fNormalizedShape;  // dims [fAxis .. fSize-1]
         auto normInner = UTILITY::ComputeStrideFromShape(normShape);

         op += SP + SP + SP + "bool const in_range = (tid < " + nl + "u);\n";
         op += SP + SP + SP + "std::size_t norm_offset = 0u;\n";
         op += SP + SP + SP + "std::size_t s_norm_offset = 0u;\n";
         if (!fNB.empty())
            op += SP + SP + SP + "std::size_t b_norm_offset = 0u;\n";
         op += SP + SP + SP + "if (in_range) {\n";

         if (fSize - fAxis == 1) {
            // Single normalised dim — simplest case
            op += SP + SP + SP + SP + "norm_offset = tid * " + strides[fAxis].GetVal() + "u;\n";
            if (fShapeScale[fAxis].dim != 1)
               op += SP + SP + SP + SP + "s_norm_offset = tid * " + scaleStrides[fAxis].GetVal() + "u;\n";
            if (!fNB.empty() && fShapeB[fAxis].dim != 1)
               op += SP + SP + SP + SP + "b_norm_offset = tid * " + biasStrides[fAxis].GetVal() + "u;\n";
         } else {
            // Multi-dim normalised space
            op += SP + SP + SP + SP + "std::size_t norm_rem = tid;\n";
            for (size_t j = fAxis; j < fSize; ++j) {
               size_t ji = j - fAxis;
               op += SP + SP + SP + SP + "{ std::size_t nj = norm_rem / " + normInner[ji].GetVal() + "u;"
                  + " norm_rem %= " + normInner[ji].GetVal() + "u;"
                  + " norm_offset += nj * " + strides[j].GetVal() + "u;";
               if (fShapeScale[j].dim != 1)
                  op += " s_norm_offset += nj * " + scaleStrides[j].GetVal() + "u;";
               if (!fNB.empty() && fShapeB[j].dim != 1)
                  op += " b_norm_offset += nj * " + biasStrides[j].GetVal() + "u;";
               op += " }\n";
            }
         }
         op += SP + SP + SP + "}\n\n";

         op += SP + SP + SP + "std::size_t const norm_idx = row_base + norm_offset;\n";
         op += SP + SP + SP + "T const val = in_range ? X[norm_idx] : static_cast<T>(0);\n\n";

         // --- Pass 1: parallel mean ---
         op += SP + SP + SP + "// Pass 1: compute mean via shared-memory tree reduction\n";
         op += SP + SP + SP + "shmem[tid] = val;\n";
         op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
         size_t half = blockSize / 2;
         while (half > 0) {
            op += SP + SP + SP + "if (tid < " + std::to_string(half) + "u) shmem[tid] += shmem[tid + " + std::to_string(half) + "u];\n";
            op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
            half >>= 1;
         }
         op += SP + SP + SP + "T const mean = shmem[0] / static_cast<T>(" + nl + ");\n";
         op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n\n";

         // --- Pass 2: parallel variance ---
         op += SP + SP + SP + "// Pass 2: compute variance\n";
         op += SP + SP + SP + "T const diff = val - mean;\n";
         op += SP + SP + SP + "shmem[tid] = in_range ? diff * diff : static_cast<T>(0);\n";
         op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
         half = blockSize / 2;
         while (half > 0) {
            op += SP + SP + SP + "if (tid < " + std::to_string(half) + "u) shmem[tid] += shmem[tid + " + std::to_string(half) + "u];\n";
            op += SP + SP + SP + "alpaka::syncBlockThreads(acc);\n";
            half >>= 1;
         }
         op += SP + SP + SP + "T const invStdDev = static_cast<T>(1) / sqrt(acc,"
               " shmem[0] / static_cast<T>(" + nl + ") + static_cast<T>(" + eps + "));\n\n";

         // Save mean/invstd if requested
         if (!fNMean.empty())
            op += SP + SP + SP + "if (tid == 0u) out_mean[row] = mean;\n";
         if (!fNInvStdDev.empty())
            op += SP + SP + SP + "if (tid == 0u) out_invstd[row] = invStdDev;\n";
         op += "\n";

         // --- Pass 3: normalise, scale, bias ---
         op += SP + SP + SP + "// Pass 3: normalize + scale + bias\n";
         op += SP + SP + SP + "if (in_range) {\n";

         // scale base (axis contribution)
         op += SP + SP + SP + SP + "std::size_t const scale_base =\n";
         {
            bool any = false;
            for (size_t i = 0; i < fAxis; ++i) {
               if (fShapeScale[i].dim != 1) {
                  op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i)
                     + " * " + scaleStrides[i].GetVal() + "u";
                  if (any) op += " +\n";
                  any = true;
               }
            }
            if (!any) op += SP + SP + SP + SP + SP + "0u";
            op += ";\n";
         }
         op += SP + SP + SP + SP + "T out_val = scale[scale_base + s_norm_offset] * invStdDev * (val - mean);\n";

         if (!fNB.empty()) {
            op += SP + SP + SP + SP + "std::size_t const bias_base =\n";
            bool any = false;
            for (size_t i = 0; i < fAxis; ++i) {
               if (fShapeB[i].dim != 1) {
                  op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i)
                     + " * " + biasStrides[i].GetVal() + "u";
                  if (any) op += " +\n";
                  any = true;
               }
            }
            if (!any) op += SP + SP + SP + SP + SP + "0u";
            op += ";\n";
            op += SP + SP + SP + SP + "out_val += bias[bias_base + b_norm_offset];\n";
         }

         // Distinguishes "this path did not run" from a stale linked body; off unless
         // SOFIE_LAYERNORM_TRACE is set.
         if (std::getenv("SOFIE_LAYERNORM_TRACE")) {
            std::fprintf(stderr, "[LN-TRACE] %s parallel-path epilogue, grid=%s\n", kname.c_str(),
                         fFakeQuantGrid ? "SET" : "unset");
         }

         if (fCarrierGrid) {
            op += EncodeToGridStatements("Y[norm_idx]", "out_val", *fCarrierGrid,
                                         SP + SP + SP + SP);
         } else if (fFakeQuantGrid) {
            op += FakeQuantRoundTripStatements("out_snapped", "out_val", *fFakeQuantGrid,
                                               SP + SP + SP + SP);
            op += SP + SP + SP + SP + "Y[norm_idx] = static_cast<T>(out_snapped);\n";
         } else {
            op += SP + SP + SP + SP + "Y[norm_idx] = out_val;\n";
         }
         op += SP + SP + SP + "}\n";  // end in_range

      } else {
         // Serial path (dynamic shapes or normalizedLength > 1024): one thread per row,
         // explicit loops over the normalized dims.
         op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (global_thread_idx >= axesLength) return;\n";
         op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";
         op += SP + SP + SP + "for (std::size_t row = global_thread_idx; row < axesLength; row += grid_thread_extent) {\n\n";

         if (fAxis > 0) {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + "std::size_t const axis_" + std::to_string(i)
                  + " = (row / " + axesStrides[i].GetVal() + "u) % "
                  + inputShape[i] + "u;\n";
            }
            op += "\n";
         }

         op += SP + SP + SP + SP + "std::size_t const row_base =\n";
         if (fAxis == 0) {
            op += SP + SP + SP + SP + SP + "0u;\n\n";
         } else {
            for (size_t i = 0; i < fAxis; ++i) {
               op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i)
                  + " * " + strides[i].GetVal() + "u";
               op += (i + 1 < fAxis) ? " +\n" : ";\n\n";
            }
         }

         op += SP + SP + SP + SP + "std::size_t const scale_base =\n";
         { bool any = false;
           for (size_t i = 0; i < fAxis; ++i) {
              if (fShapeScale[i].dim != 1) {
                 op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i)
                    + " * " + scaleStrides[i].GetVal() + "u";
                 if (any) op = " +\n" + op; any = true;
              }
           }
           if (!any) op += SP + SP + SP + SP + SP + "0u";
           op += ";\n\n";
         }

         if (!fNB.empty()) {
            op += SP + SP + SP + SP + "std::size_t const bias_base =\n";
            bool any = false;
            for (size_t i = 0; i < fAxis; ++i) {
               if (fShapeB[i].dim != 1) {
                  op += SP + SP + SP + SP + SP + "axis_" + std::to_string(i)
                     + " * " + biasStrides[i].GetVal() + "u";
                  if (any) op = " +\n" + op; any = true;
               }
            }
            if (!any) op += SP + SP + SP + SP + SP + "0u";
            op += ";\n\n";
         }

         op += SP + SP + SP + SP + "T mean = static_cast<T>(0);\n";
         for (size_t j = fAxis; j < fSize; ++j)
            op += SP + SP + SP + SP + "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + inputShape[j]
               + "u; n_" + std::to_string(j) + "++) {\n";
         op += SP + SP + SP + SP + SP + "std::size_t const norm_idx = row_base";
         for (size_t j = fAxis; j < fSize; ++j)
            op += " + n_" + std::to_string(j) + " * " + strides[j].GetVal() + "u";
         op += ";\n";
         op += SP + SP + SP + SP + SP + "mean += X[norm_idx];\n";
         for (size_t j = fAxis; j < fSize; ++j) op += SP + SP + SP + SP + "}\n";
         op += SP + SP + SP + SP + "mean /= static_cast<T>(" + fNormalizedLength + ");\n\n";

         op += SP + SP + SP + SP + "T sum = static_cast<T>(0);\n";
         for (size_t j = fAxis; j < fSize; ++j)
            op += SP + SP + SP + SP + "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + inputShape[j]
               + "u; n_" + std::to_string(j) + "++) {\n";
         op += SP + SP + SP + SP + SP + "std::size_t const norm_idx = row_base";
         for (size_t j = fAxis; j < fSize; ++j)
            op += " + n_" + std::to_string(j) + " * " + strides[j].GetVal() + "u";
         op += ";\n";
         op += SP + SP + SP + SP + SP + "T tmp = X[norm_idx] - mean;\n";
         op += SP + SP + SP + SP + SP + "sum += tmp * tmp;\n";
         for (size_t j = fAxis; j < fSize; ++j) op += SP + SP + SP + SP + "}\n";
         op += SP + SP + SP + SP + "T const invStdDev = static_cast<T>(1) / "
            "SOFIE_DEVICE_sqrt(acc, sum / static_cast<T>(" + fNormalizedLength
            + ") + static_cast<T>(" + std::to_string(fAttrEpsilon) + "));\n\n";

         if (!fNMean.empty())
            op += SP + SP + SP + SP + "out_mean[row] = mean;\n";
         if (!fNInvStdDev.empty())
            op += SP + SP + SP + SP + "out_invstd[row] = invStdDev;\n";
         op += "\n";

         for (size_t j = fAxis; j < fSize; ++j)
            op += SP + SP + SP + SP + "for (std::size_t n_" + std::to_string(j)
               + " = 0; n_" + std::to_string(j) + " < " + inputShape[j]
               + "u; n_" + std::to_string(j) + "++) {\n";
         op += SP + SP + SP + SP + SP + "std::size_t const norm_idx = row_base";
         for (size_t j = fAxis; j < fSize; ++j)
            op += " + n_" + std::to_string(j) + " * " + strides[j].GetVal() + "u";
         op += ";\n";
         op += SP + SP + SP + SP + SP + "std::size_t const s_idx = scale_base";
         for (size_t j = fAxis; j < fSize; ++j) {
            if (fShapeScale[j].dim != 1)
               op += " + n_" + std::to_string(j) + " * " + scaleStrides[j].GetVal() + "u";
         }
         op += ";\n";
         op += SP + SP + SP + SP + SP + "T val = scale[s_idx] * invStdDev * (X[norm_idx] - mean);\n";
         if (!fNB.empty()) {
            op += SP + SP + SP + SP + SP + "std::size_t const b_idx = bias_base";
            for (size_t j = fAxis; j < fSize; ++j) {
               if (fShapeB[j].dim != 1)
                  op += " + n_" + std::to_string(j) + " * " + biasStrides[j].GetVal() + "u";
            }
            op += ";\n";
            op += SP + SP + SP + SP + SP + "val += bias[b_idx];\n";
         }
         // The serial fallback applies the same snap and encode as the parallel path.
         if (fCarrierGrid) {
            op += EncodeToGridStatements("Y[norm_idx]", "val", *fCarrierGrid,
                                         SP + SP + SP + SP + SP);
         } else if (fFakeQuantGrid) {
            op += FakeQuantRoundTripStatements("val_snapped", "val", *fFakeQuantGrid,
                                               SP + SP + SP + SP + SP);
            op += SP + SP + SP + SP + SP + "Y[norm_idx] = static_cast<T>(val_snapped);\n";
         } else {
            op += SP + SP + SP + SP + SP + "Y[norm_idx] = val;\n";
         }
         for (size_t j = fAxis; j < fSize; ++j) op += SP + SP + SP + SP + "}\n";

         op += SP + SP + SP + "}\n";  // end row loop
      }

      op += SP + SP + "}\n";    // end operator()
      op += SP + "};\n";        // end struct
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "LayerNormKernel_" + opName;
      return SP + kname + " layerNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("TMVA::SOFIE LayerNormalization called to Generate without being initialized first");

      std::string axesLengthStr = fAxesLength;
      std::string kname = "layerNormKernel_" + opName;

      // Determine parallel vs serial (same logic as kernel generation)
      size_t normLenVal2 = 0;
      bool canParallel2 = false;
      try { normLenVal2 = std::stoul(fNormalizedLength); canParallel2 = (normLenVal2 > 0 && normLenVal2 <= 1024); }
      catch (...) {}
      size_t blockSize2 = 1;
      if (canParallel2) { while (blockSize2 < normLenVal2) blockSize2 <<= 1; }

      std::string args =
         "alpaka::getPtrNative(deviceBuf_" + fNX + "), "
         + "alpaka::getPtrNative(deviceBuf_" + fNScale + ")";
      if (!fNB.empty())
         args += ", alpaka::getPtrNative(deviceBuf_" + fNB + ")";
      if (!fNMean.empty())
         args += ", alpaka::getPtrNative(deviceBuf_" + fNMean + ")";
      if (!fNInvStdDev.empty())
         args += ", alpaka::getPtrNative(deviceBuf_" + fNInvStdDev + ")";
      args += ", alpaka::getPtrNative(deviceBuf_" + fNY + ")";
      args += ", static_cast<Idx>(" + axesLengthStr + ")";

      std::stringstream out;
      out << "\n//------ LAYERNORM_GPU_ALPAKA\n";
      if (canParallel2) {
         // Parallel: one block per row, blockSize2 threads per block
         out << SP << "alpaka::WorkDivMembers<Dim, Idx> workDiv_" << opName << "(\n";
         out << SP << SP << "Vec::all(Idx{" << axesLengthStr << "}),\n";    // numBlocks = rows
         out << SP << SP << "Vec::all(Idx{" << blockSize2 << "u}),\n";     // threads/block
         out << SP << SP << "Vec::all(Idx{1u}));\n";
      } else {
         // Serial fallback: normal sofie_workdiv
         out << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << axesLengthStr << "});\n";
         out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      }
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << opName
         << ", " << kname << ", " << args << ");\n";

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return { std::string("Axpy") }; }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE

#endif
