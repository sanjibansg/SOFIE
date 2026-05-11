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
   EActivationType fActivation;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeScale;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeMean;
   std::vector<size_t> fShapeVar;
   std::vector<size_t> fShapeY;

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

      if(std::is_same<T, float>::value){
      fType = "float";
      }
      else{
	      throw
		      std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a BatchNormalization operator");
      }
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType out = input[0];
      return {out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() != 5 ) {
         throw
         std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference need 5 input tensors");
      }
      for(size_t i = 0; i < input.size(); i++) {
         if (input[i].size() != 4) {
            throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op Shape inference only accept tensor with 4 dimensions");
         }
      }

      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNX + " fnx is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNScale)) {
	     throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNScale + " fns is not found in model");
      }
	  if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNB + " fnb is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNMean)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNMean + " fnm is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNVar)) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization op Input Tensor " + fNVar + " fnv is not found in model");
      }

      fShapeX = model.GetTensorShape(fNX);

      if (fShapeX.size() <  2 || fShapeX.size() > 4) {
         throw
            std::runtime_error("TMVA SOFIE BatchNormalization Op input tensor " + fNX + " fnx has wrong shape : " + ConvertShapeToString(fShapeX));
      }

      fShapeScale = model.GetTensorShape(fNScale);
      fShapeB = model.GetTensorShape(fNB);
      fShapeMean = model.GetTensorShape(fNMean);
      fShapeVar = model.GetTensorShape(fNVar);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      if (fShapeB.size() == 1) {
         // Broadcast scale, bias, input_mean and input_var to shape_X
         auto original_B = model.GetInitializedTensorData(fNB);
         auto original_S = model.GetInitializedTensorData(fNScale);
         auto original_M = model.GetInitializedTensorData(fNMean);
         auto original_V = model.GetInitializedTensorData(fNVar);
         size_t batchSize = fShapeX[0];
         size_t channels = fShapeX[1];
         size_t height = (fShapeX.size() > 2) ? fShapeX[2] : 1;
         size_t width = (fShapeX.size() > 3) ? fShapeX[3] : 1;
         size_t n = batchSize * channels * height * width;
         if (fType == "float") {
            float *original_bias = static_cast<float *>(original_B.get());
            float *original_scale = static_cast<float *>(original_S.get());
            float *original_mean = static_cast<float *>(original_M.get());
            float *original_var = static_cast<float *>(original_V.get());
            float *new_bias = new float[n];
            float *new_scale = new float[n];
            float *new_mean = new float[n];
            float *new_var = new float[n];
            size_t bs = 0, ch = 0, h = 0, w = 0;
            for (ch = 0; ch < channels; ch++) {
               for (h = 0; h < height; h++) {
                  for (w = 0; w < width; w++) {
                     new_bias[bs * channels * height * width + ch * height * width + h * width + w] = original_bias[ch];
                     new_scale[bs * channels * height * width + ch * height * width + h * width + w] =
                        original_scale[ch];
                     new_mean[bs * channels * height * width + ch * height * width + h * width + w] = original_mean[ch];
                     new_var[bs * channels * height * width + ch * height * width + h * width + w] = original_var[ch];
                  }
               }
            }
            size_t Batchoffset = channels * height * width;
            for (bs = 1; bs < batchSize; bs++) {
               std::copy(new_bias, new_bias + Batchoffset, new_bias + (bs * Batchoffset));
               std::copy(new_scale, new_scale + Batchoffset, new_scale + (bs * Batchoffset));
               std::copy(new_mean, new_mean + Batchoffset, new_mean + (bs * Batchoffset));
               std::copy(new_var, new_var + Batchoffset, new_var + (bs * Batchoffset));
            }
            //// new_var =1. / sqrt(input_var + fepsilon)
            for (size_t i = 0; i < n; i++) {
               new_var[i] = 1. / sqrt(new_var[i] + fepsilon);
               new_scale[i] *= new_var[i]; // include var in new scale
            }
            std::vector<size_t> new_bias_shape = {batchSize, channels, height, width};
            std::shared_ptr<void> new_bias_ptr(new_bias, std::default_delete<float[]>());
            std::shared_ptr<void> new_scale_ptr(new_scale, std::default_delete<float[]>());
            std::shared_ptr<void> new_mean_ptr(new_mean, std::default_delete<float[]>());
            std::shared_ptr<void> new_var_ptr(new_var, std::default_delete<float[]>());
            model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), new_bias_shape, new_bias_ptr);
            model.UpdateInitializedTensor(fNScale, model.GetTensorType(fNScale), new_bias_shape, new_scale_ptr);
            model.UpdateInitializedTensor(fNMean, model.GetTensorType(fNMean), new_bias_shape, new_mean_ptr);
            model.UpdateInitializedTensor(fNVar, model.GetTensorType(fNVar), new_bias_shape, new_var_ptr);
            fShapeB = model.GetTensorShape(fNB);
            fShapeScale = model.GetTensorShape(fNScale);
            fShapeMean = model.GetTensorShape(fNMean);
            fShapeVar = model.GetTensorShape(fNVar);
         }
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()){
         throw std::runtime_error("TMVA SOFIE Batch Normalization called to Generate without being initialized first");
      }

      std::stringstream out;
      //// Batch Norm op
      size_t batchSize = fShapeX[0];
      size_t channels = fShapeX[1];
      size_t height = (fShapeX.size() > 2) ? fShapeX[2] : 1;
      size_t width = (fShapeX.size() > 3) ? fShapeX[3] : 1;
      size_t n = batchSize * channels * height * width;

      //// copy X into Y
      out << "\n\n//---- BatchNorm\n";
      out << SP << "constexpr int " << OpName << "_N =" << batchSize * channels * height * width << ";\n";
      out << SP << "constexpr int "<<OpName<< "_incx = 1;\n";
      out << SP << "constexpr int "<<OpName<< "_incy = 1;\n";
      out << SP << "BLAS::scopy_(&" << OpName << "_N, " << "tensor_" << fNX << ", &" << OpName << "_incx," << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";

      //// blas saxpy (Y = -Bmean + Y)
      out << SP << "float "<<OpName<< "_alpha = -1;\n";
      out << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNMean << ", &" << OpName << "_incx,"
         << "tensor_" << fNY <<", &" << OpName << "_incy);\n\n ";

      //// Y *= scale*var
      out << SP << "for (size_t i = 0; i < " << n << "; i++) {\n";
      // scale tensor contains already the var
      out << SP << SP << "tensor_" << fNY << "[i] *= tensor_" << fNScale << "[i]; \n";
      out << SP << "}\n";

      //// blas saxpy (Y = Bbias + Y)
      out << SP <<OpName<< "_alpha = 1;\n";
      out << SP << "BLAS::saxpy_(&" << OpName << "_N, &" << OpName << "_alpha, " << "tensor_" << fNB << ", &" << OpName << "_incx, "
         << "tensor_" << fNY << ", &" << OpName << "_incy);\n\n";

      if(fActivation == EActivationType::RELU){
         out << SP << "for (int id = 0; id < " << ConvertShapeToLength(fShapeY) << " ; id++){\n";
         out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNY << "[id] > 0 )? tensor_" << fNY << "[id] : 0);\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty())
         throw std::runtime_error("TMVA SOFIE BatchNormalization called to Generate without being initialized first");

      std::size_t totalElements = ConvertShapeToLength(fShapeY);

      std::string kname = "BatchNormKernel_" + opName;
      std::string op;
      op  = "\n//------ BATCHNORM_KERNEL_ALPAKA\n";
      op += SP + "struct " + kname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ X,\n";
      op += SP + SP + SP + "T const* __restrict__ scale,\n";
      op += SP + SP + SP + "T const* __restrict__ bias,\n";
      op += SP + SP + SP + "T const* __restrict__ mean,\n";
      op += SP + SP + SP + "T* __restrict__ Y,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t i = global_thread_idx; i < totalElements; i += grid_thread_extent) {\n";

      op += SP + SP + SP + SP + "T val = (X[i] - mean[i]) * scale[i] + bias[i];\n";

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
         throw std::runtime_error("TMVA SOFIE BatchNormalization called to Generate without being initialized first");

      std::size_t totalElements = ConvertShapeToLength(fShapeY);
      std::string kname = "batchNormKernel_" + opName;

      std::stringstream out;
      out << "\n//------ BATCHNORM_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNY << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"   << fNY << " = Vec::all(Idx{" << totalElements << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNY
         << " = {elementsPerGrid_" << fNY << ", elementsPerThread_" << fNY << "};\n";
      out << SP << "auto const workDiv_" << fNY << " = alpaka::getValidWorkDiv(kernelCfg_" << fNY
         << ", devAcc, " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX     << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNScale << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNB     << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNMean  << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY     << ")"
         << ", static_cast<Idx>(" << totalElements << "));\n";
      
      out << SP << "auto task_" << fNY << " = alpaka::createTaskKernel<Acc>(workDiv_" << fNY
         << ", " << kname
         << ", alpaka::getPtrNative(deviceBuf_" << fNX     << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNScale << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNB     << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNMean  << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << fNY     << ")"
         << ", static_cast<Idx>(" << totalElements << "));\n";
      out << SP <<"alpaka::enqueue(queue, task_" << fNY << ");\n";
      
      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return { std::string("Copy"), std::string("Axpy") }; }
};

}//SOFIE


#endif //SOFIE_ROPERATOR_BatchNormalization
