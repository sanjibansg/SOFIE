#ifndef SOFIE_ROPERATOR_BatchNormalization
#define SOFIE_ROPERATOR_BatchNormalization

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"


#include <cmath>
#include <sstream>

namespace SOFIE{

template <typename T>
class ROperator_BatchNormalization final : public ROperator
{

private:

   /* Attributes */
   float fepsilon = 1e-05;
   float fmomentum = 0.9;
   std::size_t ftraining_mode = 0;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNMean;
   std::string fNVar;
   std::string fNY;
   std::string fNFusedScale;   // scale/sqrt(var+eps) fused over channels, shape [C]
   EActivationType fActivation;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

   std::string fType;

public:
   ROperator_BatchNormalization() = delete;

   /* Constructor */
   ROperator_BatchNormalization( float epsilon, float momentum, std::size_t training_mode,
   std::string nameX, std::string nameScale, std::string nameB,
   std::string nameMean, std::string nameVar, std::string nameY, EActivationType activation=EActivationType::UNDEFINED):
   fepsilon(epsilon), fmomentum(momentum), ftraining_mode(training_mode),
   fNX(UTILITY::Clean_name(nameX)), fNScale(UTILITY::Clean_name(nameScale)),
   fNB(UTILITY::Clean_name(nameB)), fNMean(UTILITY::Clean_name(nameMean)),
   fNVar(UTILITY::Clean_name(nameVar)), fNY(UTILITY::Clean_name(nameY)), fActivation(activation)
   {
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };

      // fused per-channel scale tensor (created in Initialize)
      fNFusedScale = fNScale + "_fused_inv_std_dev";

      if(std::is_same<T, float>::value){
      fType = "float";
      }
      else{
	      throw
		      std::runtime_error("SOFIE Encountered unsupported type parsing a BatchNormalization operator");
      }
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType out = input[0];
      return {out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() != 5 ) {
         throw
         std::runtime_error("SOFIE BatchNormalization Op Shape inference need 5 input tensors");
      }
      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("SOFIE BatchNormalization op Input Tensor " + fNX + " is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNScale)) {
         throw std::runtime_error("SOFIE BatchNormalization op Input Tensor " + fNScale + " is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error("SOFIE BatchNormalization op Input Tensor " + fNB + " is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNMean)) {
         throw std::runtime_error("SOFIE BatchNormalization op Input Tensor " + fNMean + " is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNVar)) {
         throw std::runtime_error("SOFIE BatchNormalization op Input Tensor " + fNVar + " is not found in model");
      }

      fShapeX = model.GetDimTensorShape(fNX);
      // Rank is kept at 2-4, matching ROOT SOFIE's BatchNorm constraint
      if (fShapeX.size() < 2 || fShapeX.size() > 4) {
         throw std::runtime_error("SOFIE BatchNormalization Op input tensor " + fNX
                                  + " has wrong shape : " + ConvertDimShapeToString(fShapeX));
      }

      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      // Fuse scale with the variance over channels only -> a [C] array, instead of materializing
      // the weights to the full [N,C,...] tensor (which would bake the batch size and block any dynamic shape).
      //The batch and spatial dims are handled by the runtime index math in the
      // kernel / Generate.
      auto original_S = model.GetInitializedTensorData(fNScale);
      auto original_V = model.GetInitializedTensorData(fNVar);
      auto shape_S = model.GetTensorShape(fNScale);
      if (shape_S.size() != 1) {
         throw std::runtime_error("SOFIE BatchNormalization 'scale' tensor must be 1D (per-channel).");
      }
      size_t channels = shape_S[0];

      //TODO: only float is fused here (mirrors ROOT); add a double branch if needed
      if (fType == "float") {
         float *original_scale_ptr = static_cast<float *>(original_S.get());
         float *original_var_ptr   = static_cast<float *>(original_V.get());
         float *fused_scale_data   = new float[channels];
         for (size_t i = 0; i < channels; i++) {
            // scale * (1 / sqrt(variance + epsilon))
            fused_scale_data[i] = original_scale_ptr[i] / std::sqrt(original_var_ptr[i] + fepsilon);
         }
         std::shared_ptr<void> fused_scale_ptr(fused_scale_data, std::default_delete<float[]>());
         model.AddInitializedTensor(fNFusedScale, model.GetTensorType(fNScale), {channels}, fused_scale_ptr);
      }
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty()){
         throw std::runtime_error("SOFIE Batch Normalization called to Generate without being initialized first");
      }

      std::stringstream out;
      auto batchSize = fShapeX[0].GetVal();
      auto channels  = fShapeX[1].GetVal();
      std::string spatial_dim = "1";
      if (fShapeX.size() > 2) {
         auto spatialShape = fShapeX;
         spatialShape.erase(spatialShape.begin(), spatialShape.begin() + 2);
         spatial_dim = ConvertDimShapeToLength(spatialShape);
      }

      // Per-channel affine over a [N, C, spatial] tensor. Weights stay [C] and are indexed by the
      // channel c, so batch and spatial can be any (runtime) size.
      out << "\n\n//---- BatchNorm" << (fActivation == EActivationType::RELU ? " + ReLU " : " ") << opName << "\n";
      out << SP << "{\n";
      out << SP << "   size_t i = 0;\n";
      out << SP << "   for (size_t n = 0; n < " << batchSize << "; ++n) {\n";
      out << SP << "      for (size_t c = 0; c < " << channels << "; ++c) {\n";
      out << SP << "         const float mean_val = tensor_" << fNMean << "[c];\n";
      out << SP << "         const float fused_scale_val = tensor_" << fNFusedScale << "[c];\n";
      out << SP << "         const float bias_val = tensor_" << fNB << "[c];\n";
      out << SP << "         for (size_t sp = 0; sp < " << spatial_dim << "; ++sp) {\n";
      out << SP << "            float val = (tensor_" << fNX << "[i] - mean_val) * fused_scale_val + bias_val;\n";
      if (fActivation == EActivationType::RELU) {
         out << SP << "            tensor_" << fNY << "[i] = (val > 0.0f) ? val : 0.0f;\n";
      } else {
         out << SP << "            tensor_" << fNY << "[i] = val;\n";
      }
      out << SP << "            i++;\n";
      out << SP << "         }\n";
      out << SP << "      }\n";
      out << SP << "   }\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE BatchNormalization called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string channels      = fShapeX[1].GetVal();   // per-channel weight count (static)

      // spatial_dim = product of dims after [N, C]; "1" for a rank-2 input.
      std::string spatial_dim = "1";
      std::vector<Dim> spatialShape;
      if (fShapeX.size() > 2) {
         spatialShape.assign(fShapeX.begin() + 2, fShapeX.end());
         spatial_dim = ConvertDimShapeToLength(spatialShape);
      }
      // symbolic dims inside spatial_dim -> passed as size_t kernel args (kernel-arg convention)
      std::vector<std::string> dynParams;
      for (auto &d : spatialShape)
         if (d.isParam) {
            bool seen = false;
            for (auto &q : dynParams) if (q == d.param) seen = true;
            if (!seen) dynParams.push_back(d.param);
         }

      std::string kname = "BatchNormKernel_" + opName;
      std::string op;
      op  = "\n//------ BATCHNORM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T const* __restrict__ fused_scale,\n";
      op += SP + SP + SP + "T const* __restrict__ bias,\n";
      op += SP + SP + SP + "T const* __restrict__ mean,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      for (auto &p : dynParams)
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t i = global_thread_idx; i < totalElements; i += grid_thread_extent) {\n";
      // weights are per-channel [C]; derive the channel from the flat index (layout [N, C, spatial]).
      op += SP + SP + SP + SP + "std::size_t const c = (i / (" + spatial_dim + ")) % (" + channels + ");\n";
      op += SP + SP + SP + SP + "T val = (X[i] - mean[c]) * fused_scale[c] + bias[c];\n";
      if (fActivation == EActivationType::RELU)
         op += SP + SP + SP + SP + "Y[i] = val > static_cast<T>(0) ? val : static_cast<T>(0);\n";
      else
         op += SP + SP + SP + SP + "Y[i] = val;\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string kname = "BatchNormKernel_" + opName;
      return SP + kname + " batchNormKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("SOFIE BatchNormalization called to Generate without being initialized first");

      std::string totalElements = ConvertDimShapeToLength(fShapeY);
      std::string kname = "batchNormKernel_" + opName;

      // symbolic spatial dims -> passed after the buffers, matching the kernel signature order.
      std::vector<std::string> dynParams;
      if (fShapeX.size() > 2) {
         for (auto it = fShapeX.begin() + 2; it != fShapeX.end(); ++it)
            if (it->isParam) {
               bool seen = false;
               for (auto &q : dynParams) if (q == it->param) seen = true;
               if (!seen) dynParams.push_back(it->param);
            }
      }
      std::string dynArgs;
      for (auto &p : dynParams) dynArgs += ", static_cast<std::size_t>(" + p + ")";

      std::stringstream out;
      out << "\n//------ BATCHNORM_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "auto const workDiv_" << fNY << " = sofie_workdiv(elementsPerGrid_" << fNY << ");\n";

      out << SP << "auto task_" << fNY << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX          << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNFusedScale << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNB          << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNMean       << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY          << ")"
         << dynArgs
         << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP <<"alpaka::enqueue(queue, task_" << fNY << ");\n";

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return {}; }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_BatchNormalization
