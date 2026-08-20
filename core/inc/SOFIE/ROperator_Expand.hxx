#ifndef SOFIE_ROperator_Expand
#define SOFIE_ROperator_Expand

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>


namespace SOFIE{

template<typename T>
class ROperator_Expand final : public ROperator{
private:

   std::vector<Dim> fShapeX;
   std::vector<size_t> fShape;
   std::vector<Dim> fShapeY;
   std::vector<Dim> fShapeDim;

   std::string fNX;
   std::string fNShape;
   std::string fNY;
   std::string fType;

   bool fInitialized = false;
   bool fInitializedShape = false;
   bool fInitBroadcast = false;

public:
   ROperator_Expand(){}
   ROperator_Expand(std::string nameX, std::string nameShape, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNShape(UTILITY::Clean_name(nameShape)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }


   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
        throw std::runtime_error("SOFIE Expand Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.IsInitializedTensor(fNShape)) {
         fInitializedShape = true;
         int64_t *shapeData =
           static_cast<int64_t *>(model.GetInitializedTensorData(fNShape).get());
         fShape = model.GetTensorShape(fNShape);
         if (fShape.size() != 1) {
            throw std::runtime_error("TMVA::SOFIE - Expand operator shape must be a 1d tensor.");
         }
         size_t N = fShape[0];
         // what do we do if shapeData contains negative values?
         for (size_t i = 0; i < N; i++) {
            if ( shapeData[i] < 0)
               throw std::runtime_error("TMVA::SOFIE - Expand: invalid shape value " + std::to_string(shapeData[i]));
         }
         std::vector<size_t> shape(shapeData, shapeData + N);
         fShapeDim = ConvertShapeToDim(shape);
      } else if (model.IsShapeTensor(fNShape)) {
         // case input shape is a shape tensor
         fShapeDim = model.GetShapeTensorValues(fNShape);
         fInitializedShape = true;
      } else {
         // assume shape of input shape is known (size is 1)
         auto shapeOfInputShape = model.GetTensorShape(fNShape);
         fShapeDim.resize(shapeOfInputShape[0]);
         for (size_t i = 0; i < fShapeDim.size(); i++) {
            fShapeDim[i] = Dim{std::string("v_") + fNShape + "_" + std::to_string(i)};
            model.AddShapeParam(fShapeDim[i].param);
         }
      }
      // Y is the common shape of fShapeX and shape
      auto ret  = SOFIE::UTILITY::MultidirectionalBroadcastShape(fShapeX, fShapeDim);
      fShapeY = ret.second;
      fInitialized = model.IsInitializedTensor(fNX) && fInitializedShape;
      std::vector<size_t> shapeX;
      std::vector<size_t> shapeY;
      // case shape tensor and input shape are known
      if (!model.IsDynamicTensor(fNX) && !model.IsDimInputTensor(fNX) && fInitializedShape) {
         shapeX = ConvertShapeToInt(fShapeX);
         shapeY = ConvertShapeToInt(fShapeY);
         if (!UTILITY::AreSameShape(shapeX, shapeY))
            fInitBroadcast = true;
      }
      if (fInitialized) {
         // cannot have Dim initialized tensors
         assert(!shapeX.empty() && !shapeY.empty());
         // Broadcast X to the common shape shapeY
         // If X is an initialized tensor (constant)
         auto data = model.GetInitializedTensorData(fNX);
         if (fInitBroadcast) {
            std::shared_ptr<void> broadcastedData(
               UTILITY::UnidirectionalBroadcast(static_cast<T *>(data.get()), shapeX, shapeY),
               std::default_delete<T[]>());
            // Update the data and the shape of X
            model.UpdateInitializedTensor(fNX, model.GetTensorType(fNX), shapeY, broadcastedData);
            fShapeX = fShapeY;
            // need to set as a not writable tensor
            model.SetNotWritableInitializedTensor(fNX);
            data = broadcastedData;
         }
         if (fInitBroadcast || model.IsConstantTensor(fNX)) {
            fIsOutputConstant = true; // constant output in this case
            model.AddConstantTensor(fNY, model.GetTensorType(fNX), shapeY, data);
            fOutputTensorNames.pop_back();
         } else {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), shapeY);
         }
      } else {
         // // case input is not initialized
         // if (shapeX.empty() && shapeDim.empty()) {

         // }
         // if (fInitializedShape)
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      }
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      if (model.Verbose()) {
         std::cout << "Expand - input " << fNX << " shape " << ConvertDimShapeToString(fShapeX) << " --> " << fNY << " shape "
                  << ConvertDimShapeToString(fShapeY) << (fIsOutputConstant ? ConvertValuesToString(model.GetTensorData<T>(fNY)) + " (constant)" : "") << std::endl;
      }

      if (fInitializedShape && model.IsInitializedTensor(fNShape)) {
         // Shape values are fully consumed into fShapeY/fShapeDim at generation time —
         // no device buffer needed for fNShape for Heterogeneous inference
         model.SetNotWritableInitializedTensor(fNShape);
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      if (!fIsOutputConstant && fInitialized && !fInitBroadcast) {
         // shapeX and shapeY are the same in this case
         auto length = ConvertDimShapeToLength(fShapeY);
         out << "// Copying initialized tensor " << fNX << " to " << fNY << "\n";
         out << SP << "std::copy(tensor_" << fNX << ", " << "tensor_" << fNX << " + " << length << ", tensor_" << fNY << ");\n";
      }
      return out.str();
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      if (fShapeY.empty()) {
         throw std::runtime_error("SOFIE Expand Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ Expand " << opName << " --> " << ConvertDimShapeToString(fShapeY) << "\n";
      // need to declare shape parameters for non initialized shapes
      if (!fInitializedShape) {
         for (size_t i = 0; i < fShapeDim.size(); i++) {
            out << SP << "size_t " << fShapeDim[i] << " = " << "tensor_" << fNShape << "[" << i << "];\n";
         }
      }
      // No need to broadcast A if it's an initialized tensor or shapes are the same
      if (!fInitialized && fShapeX != fShapeY) {
         out << SP << "// Broadcasting uninitialized tensor " << fNX << "\n";
         out << SP << "SOFIE::UTILITY::UnidirectionalBroadcast(tensor_" << fNX << ", " << ConvertDimShapeToString(fShapeX) << ", " << ConvertDimShapeToString(fShapeY)
                   << ", tensor_"<<fNY<<");\n";
      }
      return out.str();
   }

std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    if (fInitialized) return "";

    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("SOFIE Expand Op called to Generate without being initialized first");

    // Can only generate a static kernel if all dimensions are concrete values
    auto isStatic = [](const std::vector<Dim>& shape) {
        return std::all_of(shape.begin(), shape.end(),
                           [](const Dim& d){ return !d.isParam; });
    };
    if (!isStatic(fShapeX) || !isStatic(fShapeY)) return "";

    // Check if broadcast is actually needed
    bool needsBroadcast = (fShapeX.size() != fShapeY.size());
    if (!needsBroadcast) {
        needsBroadcast = std::any_of(fShapeX.begin(), fShapeX.end(),
                          [&](const Dim& d) {
                              size_t i = &d - fShapeX.data();
                              return fShapeX[i].dim != fShapeY[i].dim;
                          });
    }
    if (!needsBroadcast) return ""; // same static shape — just a memcpy

    const std::size_t D = fShapeY.size();

    // Left-pad fShapeX with dim=1 entries to match rank of fShapeY
    std::vector<size_t> shapeX_padded(D, 1);
    size_t offset = D - fShapeX.size();
    for (size_t i = 0; i < fShapeX.size(); ++i)
        shapeX_padded[offset + i] = fShapeX[i].dim;

    std::vector<size_t> shapeY_int(D);
    for (size_t i = 0; i < D; ++i)
        shapeY_int[i] = fShapeY[i].dim;

    auto stridesX = UTILITY::ComputeStrideFromShape(shapeX_padded);
    auto stridesY = UTILITY::ComputeStrideFromShape(shapeY_int);
    std::size_t totalElements = ConvertShapeToLength(shapeY_int);

    std::string kname = "ExpandKernel_" + opName;

    std::string op;
    op  = "\n//------ EXPAND_KERNEL_ALPAKA\n";
    op += SP + "struct " + kname + " {\n";
    op += SP + SP + "template<typename TAcc, typename T>\n";
    op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
    op += SP + SP + SP + "TAcc const& acc,\n";
    op += SP + SP + SP + "T const* __restrict__ input,\n";
    op += SP + SP + SP + "T* __restrict__ output,\n";
    op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

    op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
    op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
    op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

    op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

    // Decompose output linear index using compile-time output strides
    for (std::size_t d = 0; d < D; ++d) {
        op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
            + " = (elem_idx / " + std::to_string(stridesY[d]) + "u) % "
            + std::to_string(shapeY_int[d]) + "u;\n";
    }
    op += "\n";

    // Input index: broadcast dims (shapeX_padded[d]==1) contribute 0 —
    // compiler eliminates zero terms entirely, no runtime branch
    op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
    for (std::size_t d = 0; d < D; ++d) {
        if (shapeX_padded[d] == 1) {
            op += SP + SP + SP + SP + SP + "0u";
        } else {
            op += SP + SP + SP + SP + SP
                + "out_" + std::to_string(d)
                + " * " + std::to_string(stridesX[d]) + "u";
        }
        op += (d + 1 < D) ? " +\n" : ";\n\n";
    }

    op += SP + SP + SP + SP + "output[elem_idx] = input[input_idx];\n";
    op += SP + SP + SP + "}\n";   // end grid-stride loop
    op += SP + SP + "}\n";        // end operator()
    op += SP + "};\n";            // end struct

    return op;
}

std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    if (fInitialized) return "";

    auto isStatic = [](const std::vector<Dim>& shape) {
        return std::all_of(shape.begin(), shape.end(),
                           [](const Dim& d){ return !d.isParam; });
    };
    if (!isStatic(fShapeX) || !isStatic(fShapeY)) return "";

    // Check if broadcast is actually needed
    bool needsBroadcast = (fShapeX.size() != fShapeY.size());
    if (!needsBroadcast) {
        for (size_t i = 0; i < fShapeX.size(); ++i)
            if (fShapeX[i].dim != fShapeY[i].dim) { needsBroadcast = true; break; }
    }
    if (!needsBroadcast) return "";

    opName = "op_" + opName;
    std::string kname = "ExpandKernel_" + opName;
    return SP + kname + " expandKernel_" + opName + ";\n";
}

std::string Generate_GPU_ALPAKA(std::string opName) override {
    if (fIsOutputConstant) return "";
    opName = "op_" + opName;
    if (fShapeY.empty())
        throw std::runtime_error("SOFIE Operator Expand called to Generate without being initialized first");

    std::stringstream out;
    out << "\n//------ EXPAND_GPU_ALPAKA\n";

    if (fInitialized && !fInitBroadcast) {
        // GenerateInitCode already handled the copy — nothing to do at inference time
        return "";
    }

    auto isStatic = [](const std::vector<Dim>& shape) {
        return std::all_of(shape.begin(), shape.end(),
                           [](const Dim& d){ return !d.isParam; });
    };
    bool staticShapes = isStatic(fShapeX) && isStatic(fShapeY);

    // Check if broadcast is actually needed for static shapes
    bool needsBroadcast = !staticShapes; // dynamic always needs runtime broadcast
    if (staticShapes) {
        needsBroadcast = (fShapeX.size() != fShapeY.size());
        if (!needsBroadcast) {
            for (size_t i = 0; i < fShapeX.size(); ++i)
                if (fShapeX[i].dim != fShapeY[i].dim) { needsBroadcast = true; break; }
        }
    }

    if (!needsBroadcast) {
        // Same static shape — device-to-device copy
        out << SP << "alpaka::memcpy(queue, deviceBuf_" << fNY
            << ", deviceBuf_" << fNX << ");\n";
        out << SP << "alpaka::wait(queue);\n";
        return out.str();
    }

    if (!staticShapes) {
        // Dynamic shapes — not yet supported on GPU, throw a clear error
        throw std::runtime_error(
            "SOFIE Expand GPU: dynamic shapes are not yet supported for GPU inference. "
            "Tensor " + fNX + " has a dynamic shape.");
    }

    // Static broadcast — launch the expand kernel
    std::vector<size_t> shapeY_int(fShapeY.size());
    for (size_t i = 0; i < fShapeY.size(); ++i)
        shapeY_int[i] = fShapeY[i].dim;
    std::size_t totalElements = ConvertShapeToLength(shapeY_int);
    std::string kname = "expandKernel_" + opName;

    out << SP << "auto const elementsPerThread_" << opName << " = Vec::all(static_cast<Idx>(1));\n";
    out << SP << "auto const elementsPerGrid_"   << opName << " = Vec::all(Idx{" << totalElements << "});\n";
    out << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
    out << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
        << ", " << kname
        << ", alpaka::getPtrNative(deviceBuf_" << fNX << ")"
        << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
        << ", static_cast<Idx>(" << totalElements << "));\n";
   out << SP <<"alpaka::enqueue(queue, task_" << opName << ");\n";

    return out.str();
}

EFusionMappingType GetFusionMappingType() const override {
   if (fIsOutputConstant || fInitialized)
       return EFusionMappingType::Unsupported;

   const auto isStatic = [](const std::vector<Dim> &shape) {
       return std::all_of(shape.begin(), shape.end(), [](const Dim &dim) {
          return !dim.isParam;
       });
   };

   if (!isStatic(fShapeX) || !isStatic(fShapeY))
       return EFusionMappingType::Unsupported;

   if (fShapeX == fShapeY)
       return EFusionMappingType::OneToOne;

   return EFusionMappingType::OneToMany;
}

std::string GetFusionExpr(const std::vector<std::string> &inputs) const override {
   if (inputs.size() != 1)
       return "";

   const auto mapping = GetFusionMappingType();
   if (mapping != EFusionMappingType::OneToOne && mapping != EFusionMappingType::OneToMany)
       return "";

   return inputs[0];
}

bool SupportsFusionTypes(const std::vector<ETensorType> &inputTypes, ETensorType outputType) const override
{
   return inputTypes.size() == 1 && inputTypes[0] == outputType;
}
};
}//SOFIE

#endif //SOFIE_ROperator_Expand
