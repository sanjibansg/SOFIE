#ifndef SOFIE_ROPERATOR_RANGE
#define SOFIE_ROPERATOR_RANGE

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <algorithm>

namespace SOFIE{

template <typename T>
class ROperator_Range final : public ROperator
{
private:

   std::string fNStart;
   std::string fNLimit;
   std::string fNDelta;
   std::string fNOutput;
   std::vector<Dim> fShape;
   std::string fType;

   // element count computed at run time from the three scalar inputs, read through host
   // pointers named tensor_<input>; shared by the CPU loop and the GPU launch
   std::string RuntimeSizeExpr() const {
      return "static_cast<size_t>(std::max(std::ceil((static_cast<float>(*tensor_" + fNLimit +
             ") - static_cast<float>(*tensor_" + fNStart + ")) / static_cast<float>(*tensor_" + fNDelta + ")), 0.0f))";
   }

public:
   ROperator_Range(){}

   ROperator_Range(std::string start, std::string limit, std::string delta, std::string nameOutput):
      fNStart(start), fNLimit(limit), fNDelta(delta),
      fNOutput(UTILITY::Clean_name(nameOutput)) {
      if (std::is_same<T, float>::value) {
          fType = "float";
      } else if (std::is_same<T, int64_t>::value) {
          fType = "int64_t";
      }
      static_assert( (std::is_same_v<T, float> || std::is_same_v<T, int64_t>),
                  "TMVA::SOFIE - Unsupported type by Range operator");
      {
         fInputTensorNames = { fNStart, fNLimit, fNDelta };
         fOutputTensorNames = { fNOutput };
      }
   }

   void Initialize(RModel& model) override {
       //input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNStart)) {
         throw
            std::runtime_error("SOFIE Range Op Input Tensor " + fNStart + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNLimit)) {
         throw
            std::runtime_error("SOFIE Range Op Input Tensor " + fNLimit + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNDelta)) {
         throw
            std::runtime_error("SOFIE Range Op Input Tensor " + fNDelta + "is not found in model");
      }
      ETensorType type = ConvertStringToType(fType);



      auto analyzeInput = [&](const std::string & tName, T & value, Dim & dim) {
         int ftype = 0; // type of input (0 intermediate, 1 constant , 2 shape)
         if (model.IsInitializedTensor(tName)) {
            T * data = static_cast<T*>(model.GetInitializedTensorData(tName).get());
            if (!data)
               throw std::runtime_error("SOFIE Range Op Input Tensor has invalid input  data");
            value = *data;
            ftype = 1;
         } else if (model.IsShapeTensor(tName)) {
            auto data = model.GetShapeTensorValues(tName);
            dim = data[0];
            if (!dim.isParam) {
               value = static_cast<T>(dim.dim);
               ftype = 1;
            } else
               ftype = 2;
         }
         return ftype;
      };

      T start_value{};
      T limit_value{};
      T delta_value{};
      Dim start_dim{};
      Dim limit_dim{};
      Dim delta_dim{};
      int res1 = analyzeInput(fNStart, start_value, start_dim);
      int res2 = analyzeInput(fNLimit, limit_value, limit_dim);
      int res3 = analyzeInput(fNDelta, delta_value, delta_dim);
      if (res1 == 0 || res2 == 0 || res3 == 0) {
         // cannot know at compile time- need to do fully at run time
         //
         fShape = {Dim{"range_size_" + fNStart + "_" + fNLimit}};
         model.AddDynamicTensor(fNOutput, type, fShape);
      } else if (res1 == 1 && res2 == 1 && res3 == 1) {
         size_t number_of_elements = std::max(static_cast<int>(std::ceil((limit_value - start_value) / delta_value )) , 0 );
         fIsOutputConstant = true;

         // compute output
         std::vector<T> output(number_of_elements);
         for (size_t i=0; i<number_of_elements; ++i) {
            output[i] =  start_value + (i * delta_value);
         }
         std::vector<size_t> shape = {number_of_elements};
         model.AddConstantTensor(fNOutput,shape, output.data());
         fShape = ConvertShapeToDim(shape);

      } else { // case of a shape tensor
         std::string start = (res1 == 1) ? std::to_string(start_value) : start_dim.GetVal();
         std::string limit = (res2 == 1) ? std::to_string(limit_value) : limit_dim.GetVal();
         std::string delta = (res3 == 1) ? std::to_string(delta_value) : delta_dim.GetVal();
         std::stringstream s;
         if (type == ETensorType::FLOAT ) {
            if (delta_value == 1)
               s <<  "std::max(std::ceil("<< limit << " - " << start << "),0.0f)";
            else
               s <<  "std::max(std::ceil(("<< limit << " - " << start << ")/" << delta << "),0.0f)";
         } else if (type == ETensorType::INT64 ) {
            if (delta == "1") {
               if (start == "0")
                  s <<  limit;
               else
                  s << "std::max((" << limit << " - " << start << "),0L)";
            } else {
               if (start == "0")
                  s <<  "((" << limit << ")/" << delta << ")";
               else
                  s << "std::max((" << limit << " - " << start << ")/"<< delta << "),0L)";
            }
         } else {
            throw
               std::runtime_error("SOFIE Range Op Input Tensor " + ConvertTypeToString(type) + "is not supported");
         }


         fShape = { Dim {s.str(), static_cast<size_t>(-1)} };
         model.AddDynamicTensor(fNOutput,type, fShape);
      }


      if (model.Verbose()) {
         std::cout << "Range -> output is " << fNOutput << " : " << ConvertDimShapeToString(fShape);
         if (fIsOutputConstant) std::cout << " : " << ConvertValuesToString(model.GetTensorData<T>(fNOutput));
         std::cout << std::endl;
      }
   }

   std::string Generate(std::string opName) override {

      std::stringstream out;
      out << "\n//------ Range " << opName << "---> " << ConvertDimShapeToString(fShape) << "\n";
      if (fIsOutputConstant) return out.str();

      opName = "op_" + opName;
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Range operator called to Generate without being initialized first");
      }

      std::string outputSizeVar;
      std::string outputSize = fShape[0].param;
      if (outputSize.find("range_size") != std::string::npos) {
         outputSizeVar = outputSize;
         outputSize = RuntimeSizeExpr();
      } else {
         outputSizeVar = "range_" + opName;
      }
      out << SP << "size_t " << outputSizeVar <<  " = " << outputSize << ";\n";
      out << SP << "for (size_t i = 0; i < " << outputSizeVar << "; i++) {\n";
      out << SP << SP << "tensor_" << fNOutput << "[i] = *tensor_" << fNStart << " + i * (*tensor_" << fNDelta << ");\n";
      out << SP << "}\n";

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      std::string op;
      op += "\n//------ RANGE_KERNEL_ALPAKA\n";
      op += SP + "struct RangeKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* start, T const* delta, T* output, std::size_t n) const {\n";
      op += SP + SP + SP + "auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (idx >= n) return;\n";
      op += SP + SP + SP + "output[idx] = start[0] + static_cast<T>(idx) * delta[0];\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (fIsOutputConstant) return "";
      opName = "op_" + opName;
      return SP + "RangeKernel_" + opName + " rangeKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      std::stringstream out;
      out << "\n//------ Range (GPU) " << opName << " ---> " << ConvertDimShapeToString(fShape) << "\n";
      if (fIsOutputConstant) return out.str();
      if (fShape.empty()) {
         throw std::runtime_error("SOFIE Range operator called to Generate without being initialized first");
      }

      opName = "op_" + opName;
      std::string outputSize = fShape[0].param;
      if (outputSize.find("range_size") != std::string::npos) {
         /*
          * Run-time size: start, limit and delta are produced by other operators, so their values
          * exist only on the device during inference. The generated code copies the three scalars
          * to the host, computes the element count with the same expression the CPU code uses, and
          * declares it under the name the downstream launches already reference. The output buffer
          * was allocated in the constructor from the value passed there under that same name, so
          * the count is checked against it before the kernel runs.
          */
         std::vector<std::string> inputs;
         for (auto &in : {fNStart, fNLimit, fNDelta}) {
            if (std::find(inputs.begin(), inputs.end(), in) == inputs.end())
               inputs.push_back(in);
         }
         std::string sizeMember = memberNameForDimShape(outputSize);

         out << SP << "size_t " << outputSize << ";\n";
         out << SP << "{\n";
         for (auto &in : inputs) {
            out << SP << SP << "auto host_" << in << " = alpaka::allocBuf<" << fType << ", Idx>(host, Ext1D::all(Idx{1}));\n";
            out << SP << SP << "alpaka::memcpy(queue, host_" << in << ", deviceBuf_" << in << ");\n";
         }
         out << SP << SP << "alpaka::wait(queue);\n";
         for (auto &in : inputs) {
            out << SP << SP << "const " << fType << "* tensor_" << in << " = alpaka::getPtrNative(host_" << in << ");\n";
         }
         out << SP << SP << outputSize << " = " << RuntimeSizeExpr() << ";\n";
         out << SP << "}\n";
         out << SP << "if (" << outputSize << " > " << sizeMember << ") {\n";
         out << SP << SP << "throw std::runtime_error(\"SOFIE Range " << opName
             << ": run-time size exceeds the size given at construction (" << outputSize << ")\");\n";
         out << SP << "}\n";
      }

      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerGrid_" << opName << " = Vec::all(Idx{" << outputSize << "});\n";
      out << SP << SP << "auto const workDiv_" << opName << " = sofie_workdiv(elementsPerGrid_" << opName << ");\n";
      out << SP << SP << "auto task_" << opName << " = alpaka::createTaskKernel<Acc>(workDiv_" << opName
          << ", rangeKernel_" << opName
          << ", alpaka::getPtrNative(deviceBuf_" << fNStart << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNDelta << ")"
          << ", alpaka::getPtrNative(deviceBuf_" << fNOutput << ")"
          << ", static_cast<std::size_t>(" << outputSize << "));\n";
      out << SP << SP << "alpaka::enqueue(queue, task_" << opName << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

}//SOFIE

#endif //SOFIE_ROPERATOR_RANGE
