#ifndef SOFIE_ROPERATOR_Concat
#define SOFIE_ROPERATOR_Concat


 #include "SOFIE/SOFIE_common.hxx"
 #include "SOFIE/ROperator.hxx"
 #include "SOFIE/RModel.hxx"

 #include <sstream>
 #include <algorithm>
 #include <iterator>
 #include <iomanip>
 #include <limits>

 namespace SOFIE{

     class ROperator_Concat final : public ROperator
     {
     private:
         int fAxis=0;
         int fnewAxis=0;
         std::vector<std::string> fInputs;
         std::string fOutput;
         std::vector<Dim>fOutputShape;
         std::vector<Dim> fOutputShapeData; // in case output is a shape tensor we store here the output shape value data (can be parametric)
         std::vector<std::vector<Dim>> fInputShapes;
         ETensorType fInputType = ETensorType::UNDEFINED;

     public:

         ROperator_Concat(){}
         ROperator_Concat(std::vector<std::string> inputs, int axis, int newAxis, std::string output):
         fAxis(axis), fnewAxis(newAxis), fOutput(UTILITY::Clean_name(output)) {
            fInputs.reserve(inputs.size());
            for (auto & name : inputs)
               fInputs.push_back(UTILITY::Clean_name(name));

         fInputTensorNames.resize(fInputs.size());
         std::transform(fInputs.begin(), fInputs.end(), fInputTensorNames.begin(),
                   [](const std::string& s) -> std::string_view { return s; });
         fOutputTensorNames = { fOutput };
         }

         std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
             return input;
         }

         // get shape of output given inputs. It is going to be called after initialized
         std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> inputs) override {
             std::vector<std::vector<size_t>> ret(1);
            // treat negative axis case
            if (fAxis<0) {
               fAxis = inputs[0].size()+fAxis;
            }
            if (fAxis < 0 || fAxis >= (int) inputs[0].size())
               throw std::runtime_error("SOFIE Concat Op - invalid axis value ");

            int concat_dim=0;
            // case of Concat (fNewAxis = 0) and not ConcatFromSequence
            if(fnewAxis == 0){
               for (size_t i = 0; i < inputs.size(); i++) {
                  if (i > 0 && inputs[i].size() != inputs[i - 1].size())
                     throw std::runtime_error("SOFIE Concat Op - input tensors have different shapes " +
                                              ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i - 1]));
                  for (size_t iaxis = 0; iaxis < inputs[i].size(); iaxis++) {
                     if ((int)iaxis == fAxis)
                        concat_dim += inputs[i][iaxis];
                     else if (i > 0 && inputs[i][iaxis] != inputs[i - 1][iaxis])
                        throw std::runtime_error("SOFIE Concat Op - input tensors have wrong shapes " +
                                                 ConvertShapeToString(inputs[i]) + " and " +
                                                 ConvertShapeToString(inputs[i - 1]));
                  }
               }

               // output shape
               ret[0] = inputs[0];
               ret[0][fAxis] = concat_dim;
            }
            std::vector<int> stack;
            // case ConCatFromSequence
            if(fnewAxis == 1){
               for(size_t i = 0; i < inputs.size(); i++) {
                  if (i > 0 && inputs[i].size() != inputs[i-1].size() )
                  throw std::runtime_error("SOFIE Concat Op - input tensors have different shapes " + fInputs[i] + " : " +
                     ConvertShapeToString(inputs[i]) + " and " + fInputs[i-1] + " : " + ConvertShapeToString(inputs[i-1]));
                  for (size_t iaxis = 0; iaxis < inputs[i].size(); iaxis++) {
                     if ((int) iaxis == fAxis)
                        stack.push_back(inputs[i][iaxis]);
                     else
                     if (i> 0 && inputs[i][iaxis] != inputs[i-1][iaxis])
                        throw std::runtime_error("SOFIE Concat Op - input tensors have wrong shapes " +
                        ConvertShapeToString(inputs[i]) + " and " + ConvertShapeToString(inputs[i-1]));
                  }

               }
               for(auto it:stack)
               ret[0].push_back(it);
            }

            return ret;
         }

         // get shape of output given inputs. It is going to be called after initialized
         std::vector<Dim> ShapeInference(const std::vector<std::vector<Dim>> & inputs, const RModel & model) {
            std::vector<Dim> ret(inputs[0].size());
            // treat negative axis case
            if (fAxis<0) {
               fAxis = inputs[0].size()+fAxis;
            }
            if (fAxis < 0 || fAxis >= (int) inputs[0].size())
               throw std::runtime_error("SOFIE Concat Op - invalid axis value ");

            Dim concat_dim;
            if(fnewAxis == 0){
               for (size_t i = 0; i < inputs.size(); i++) {
                  if (i > 0 && inputs[i].size() != inputs[i - 1].size())
                     throw std::runtime_error("SOFIE Concat Op - input tensors have different shapes " + fInputs[i] + " : " +
                                              ConvertDimShapeToString(inputs[i]) + " and " + fInputs[i-1] + " : " + ConvertDimShapeToString(inputs[i - 1]));
                  for (size_t iaxis = 0; iaxis < inputs[i].size(); iaxis++) {
                     if ((int)iaxis == fAxis) {
                        // support both integer and params shape for the concatenation axis
                        if (concat_dim.param.empty() && concat_dim.dim == 0)
                           concat_dim = inputs[i][iaxis];
                        else if (inputs[i][iaxis].isParam || concat_dim.isParam) {
                           concat_dim =
                              Dim{ concat_dim.GetVal() + std::string(" + ") + inputs[i][iaxis].GetVal(),
                                 static_cast<size_t>(-1)};
                        } else {
                           concat_dim = Dim { concat_dim.dim + inputs[i][iaxis].dim };
                        }
                     }
                     else if (i == 0) {
                        ret[iaxis] = inputs[i][iaxis];
                     }
                     else if ((!inputs[i][iaxis].isParam && !ret[iaxis].isParam) && (inputs[i][iaxis].dim != ret[iaxis].dim)) {
                        throw std::runtime_error("SOFIE Concat Op - input tensors have wrong shapes " +
                                                 ConvertDimShapeToString(inputs[i]) + " and " +
                                                 ConvertDimShapeToString(inputs[i - 1]));
                     }
                     else if (!inputs[i][iaxis].isParam && ret[iaxis].isParam){
                        // if shape is not parametric use it
                        ret[iaxis] = inputs[i][iaxis];
                     }
                     else if (inputs[i][iaxis].isParam && ret[iaxis].isParam) {
                        // check which parameter is first in RModel list
                        auto & dimNames = model.GetDimShapeNames();
                        auto p1 = std::find(dimNames.begin(), dimNames.end(), inputs[i][iaxis].param);
                        auto p2 = std::find(dimNames.begin(), dimNames.end(), ret[iaxis].param);
                        if (p1 < p2) ret[iaxis] = inputs[i][iaxis];
                     }

                  }
                  // add parenthesis in case is an expression
                  if (concat_dim.isParam && concat_dim.dim == static_cast<size_t>(-1))
                     concat_dim =  Dim{ std::string("(") + concat_dim.GetVal() +  std::string(")"), concat_dim.dim };
               }

               // output shape for concatenated axis
               ret[fAxis] = concat_dim;

            }
            // case of stacking (not supported yet)
            // here we need to check that input shapes are the same
            // for example for fAxis == 0
            // output shapes: [inputs.size(), inputs[0][0], inputs[0][1],....]
            if(fnewAxis == 1){
               throw std::runtime_error("SOFIE Concat Op - stacking (i.e. COncatFromSequence with new_axis=1) is not supported ");
            }
            return ret;
         }

         void Initialize(RModel& model) override {
            std::vector<std::vector<size_t>> inputIntShapes;
            for (auto &it : fInputs) {
               if (model.CheckIfTensorAlreadyExist(it) == false) {
                  throw std::runtime_error("SOFIE Concat Op Input Tensor " + it + " is not found in model");
               }
               fInputShapes.push_back(model.GetDimTensorShape(it));
               if (!model.IsDynamicTensor(it)) {
                  inputIntShapes.push_back(ConvertShapeToInt(fInputShapes.back()));
               }
            }
            if (inputIntShapes.size() == fInputs.size()) {
               // if all input shapes are static we can compute output shape at initialization time
               auto outputIntShape = ShapeInference(inputIntShapes)[0];
               fOutputShape = ConvertShapeToDim(outputIntShape);
               if (model.Verbose())
                  std::cout << "Initialize Concat operator with defined inputs shapes, "
                           << "output has shape " << ConvertShapeToString(outputIntShape) << std::endl;

            } else {
               // if at least one input shape is dynamic we need to compute output shape using the symbolic expression for the dimensions
               fOutputShape = ShapeInference(fInputShapes, model);
               if (model.Verbose())
                  std::cout << "Initialize Concat operator with dynamic inputs shapes, "
                           << "output has shape " << ConvertDimShapeToString(fOutputShape) << std::endl;
            }

            // check if concat has constant inputs , axis 0(concat contigous memory and type is integer)
            bool isOutputShape = false;

            // if (model.GetTensorType(fInputs[0]) == ETensorType::INT64 && fAxis == 0) {
            fIsOutputConstant = true;
            isOutputShape = true;

            for (auto &input : fInputs) {
               if (model.IsDynamicTensor(input)) {
                  fIsOutputConstant = false;
                  isOutputShape = false;
                  break;
               }
               if (!model.IsInitializedTensor(input)) {
                  if (model.IsShapeTensor(input)) {
                     // if it is a shape tensor we can have constant output if the shapes are defined)
                     auto shapeData = model.GetShapeTensorValues(input);
                     bool isShapeFullyDefined = ConvertShapeToInt(shapeData).size() == shapeData.size();
                     if (!isShapeFullyDefined) {
                        fIsOutputConstant = false;
                     } else {
                        // if shape is fully defined we can consider output as constant and we can compute the output
                        // shape at initialization time
                        fIsOutputConstant = fIsOutputConstant && true;
                     }
                     // inputs are then shape tensors and output is a shape tensor
                     isOutputShape = true;
                  } else {
                     // case of standard intermediate tensor
                     fIsOutputConstant = false;
                     isOutputShape = false;
                     break;
                  }
               } else {
                  fIsOutputConstant = fIsOutputConstant && true;
               }
            }
            //}

            if (fIsOutputConstant) {
               auto outputShape = ConvertShapeToInt(fOutputShape); // conversion must be possible
               std::vector<int64_t> outputData(ConvertShapeToLength(outputShape));
               size_t offset = 0;
               for (auto &input : fInputs) {
                  auto inputData = static_cast<int64_t *>(model.GetInitializedTensorData(input).get());
                  auto inputShape = model.GetTensorShape(input); // shape is not dynamic if it is constant
                  size_t inputLength = ConvertShapeToLength(inputShape);
                  std::copy(inputData, inputData + inputLength, outputData.begin() + offset);
                  offset += inputLength;
                  // the data of the input tensor don't need to be written in the generated code and data file
                  model.SetNotWritableInitializedTensor(input);
               }
               model.AddConstantTensor<int64_t>(fOutput, outputShape, outputData.data());
               if (model.Verbose()) {
                  std::cout << "output of Concat is a constant tensor " << ConvertShapeToString(outputShape) << " : "
                            << ConvertValuesToString(outputData) << " (constant)" << std::endl;
               }
            } else if (isOutputShape) {
               auto outputShape = ConvertShapeToInt(fOutputShape); // conversion must be possible
               if (outputShape.size() != 1)
                  throw std::runtime_error("SOFIE Concat Op - output shape for shape tensor must have rank 1");
               // output shape is a rank 1 tensor with size equal to the output rank
               std::vector<Dim> outputData(outputShape[0]);
               size_t offset = 0;
               for (auto &input : fInputs) {
                  std::vector<Dim> inputData;
                  auto inputShape = model.GetTensorShape(input);         // shape is not dynamic
                  size_t inputLength = ConvertShapeToLength(inputShape); // shape can be a scalar
                  if (model.IsShapeTensor(input)) {
                     inputData = model.GetShapeTensorValues(input);
                  } else if (model.IsInitializedTensor(input)) {
                     inputData.resize(inputLength);
                     auto intData = static_cast<int64_t *>(model.GetInitializedTensorData(input).get());
                     for (size_t i = 0; i < inputData.size(); i++)
                        inputData[i] = Dim{static_cast<size_t>(intData[i])};
                  } else {
                     // this should not happen
                     throw std::runtime_error("SOFIE Concat Operator- invalid tensor input " + input +
                                              " for shape output type");
                  }
                  std::copy(inputData.begin(), inputData.end(), outputData.begin() + offset);
                  offset += inputLength;
               }
               // add output tensor
               model.AddShapeTensor(fOutput, outputData, false); // cannot be a  scalar
               fOutputShapeData = outputData;
               if (model.Verbose()) {
                  std::cout << "output of Concat is a shape tensor " << ConvertShapeToString(outputShape) << " : "
                            << ConvertDimShapeToString(outputData) << " (shape)" << std::endl;
               }
               fIsOutputParamShape = true;
            }
            if (!fIsOutputConstant && !fIsOutputParamShape) {
               fInputType = model.GetTensorType(fInputs[0]);
               model.AddIntermediateTensor(fOutput, fInputType, fOutputShape);
               if (model.Verbose()) {
                  std::cout << "Concat ---> " << fOutput << " " <<  ConvertDimShapeToString(fOutputShape) << std::endl;
               }
            }
         }

         std::string Generate(std::string opName) override {
            opName = "op_" + opName;
            std::stringstream out;
            out<<"\n//--------- Concat " << opName << " --> " << fOutput << "  " << ConvertDimShapeToString(fOutputShape) << "\n";

            if (fIsOutputConstant) return out.str();

            if (fIsOutputParamShape) {
               // output is a shape tensor defined by the concatenation of the input shapes
               out << "// output is a shape tensor defined by the concatenation of the input shapes\n";
               for (int i = 0; i < static_cast<int>(fOutputShape
                  [0].dim); i++) {
                  out << SP << "tensor_" << fOutput << "[" << i << "] = " << fOutputShapeData[i] << ";\n";
               }
               return out.str();
            }
            // special case when memory is contiguous
            bool hasShapeOnes = true;
            for(int i = 0; i<fAxis; ++i){
               if(fInputShapes[0][i].dim !=1){
                  hasShapeOnes = false;
                  break;
               }
            }
            if (fAxis == 0 || hasShapeOnes) {
               std::string offset;
               for(size_t i=0; i<fInputs.size(); ++i) {
                  auto length = ConvertDimShapeToLength(fInputShapes[i]);
                  out << SP << "SOFIE::Copy(tensor_" << fOutput;
                  if (i > 0)
                     out << offset;
                  offset += " + " + length;
                  out << ", " << "tensor_" << fInputs[i] << ", " + length << ");\n";
               }
            }
            else {

               std::vector<Dim> outStride = UTILITY::ComputeStrideFromShape(fOutputShape);
               std::vector<std::vector<Dim>> inStrides(fInputs.size());
               int idx = 0;
               for ( auto &s : inStrides) {
                  s = UTILITY::ComputeStrideFromShape(fInputShapes[idx]);
                  idx++;
               }
               for (int i = 0; i < fAxis; ++i) {
                  // loop on dimensions
                  out << SP << "for (size_t i" << i << " = 0; i" << i << " < " << fOutputShape[i].GetVal() << "; ++i" << i <<") {\n";
               }

               out << SP << SP << SP << "int idxOut = ";
               for (int k = 0; k < fAxis; k++) {
                  if (k > 0) out << " + ";
                  out << outStride[k].GetVal() << "*i" << k;
               }
               out << ";\n";

               for (size_t j = 0; j < fInputs.size(); j++) {
                  if (j>0)
                  out << SP << SP << SP << "idxOut += " << inStrides[j-1][fAxis-1].GetVal() << ";\n";
                  out << SP << SP << SP << "int idxIn" << j <<" = ";
                  for (int k = 0; k < fAxis; k++) {
                     if (k > 0) out << " + ";
                     out << inStrides[j][k].GetVal() << "*i" << k;
                  }
                  out << ";\n";
                  out << SP << SP << SP << "for (size_t iC = 0; iC < " << inStrides[j][fAxis-1].GetVal() << "; ++iC) {\n";
                  out << SP << SP << SP << SP << "tensor_" << fOutput << "[idxOut+iC] = tensor_" << fInputs[j] << "[idxIn" << j << "+iC];\n";
                  out << SP << SP << SP << "}\n";
               // concatenate the axis values
               }
                for (int i = 0; i < fAxis; ++i) {
                    out << SP << "}\n";
                }
            }

            return out.str();
         }

   // Dynamic shape params (e.g. N, n_pf) that appear in the emitted index math and are
   // passed to the kernel as size_t args; shared by the kernel signature and the launch.
   std::vector<std::string> GetGPUDynParams() const {
      std::vector<std::string> params;
      UTILITY::CollectDimParams(UTILITY::ComputeStrideFromShape(fOutputShape), params);
      for (std::size_t k = 0; k < fInputShapes.size(); ++k) {
         UTILITY::CollectDimParams({fInputShapes[k][fAxis]}, params);
         UTILITY::CollectDimParams(UTILITY::ComputeStrideFromShape(fInputShapes[k]), params);
      }
      return params;
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      if (fIsOutputConstant || fIsOutputParamShape) return "";
      opName = "op_" + opName;
      if (fOutputShape.empty())
         throw std::runtime_error("SOFIE Operator Concat called to Generate without being initialized first");

      const std::size_t D   = fOutputShape.size();
      const std::size_t Nin = fInputs.size();

      auto outStrides = UTILITY::ComputeStrideFromShape(fOutputShape);

      // cumulative offsets along the concat axis, as strings since a dim can be symbolic
      std::vector<std::string> prefix(Nin);
      prefix[0] = "0";
      for (std::size_t k = 1; k < Nin; ++k)
         prefix[k] = prefix[k - 1] + " + (" + fInputShapes[k - 1][fAxis].GetVal() + ")";

      std::vector<std::vector<Dim>> inStrides(Nin);
      for (std::size_t k = 0; k < Nin; ++k)
         inStrides[k] = UTILITY::ComputeStrideFromShape(fInputShapes[k]);

      std::string op;
      op  = "\n//------ CONCAT_KERNEL_ALPAKA\n";
      op += SP + "struct ConcatKernel_" + opName + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "std::array<T const*, " + std::to_string(Nin) + "> inputs,\n";
      op += SP + SP + SP + "T* output,\n";
      for (auto &p : GetGPUDynParams())
         op += SP + SP + SP + "std::size_t const " + p + ",\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "std::size_t remaining;\n";
      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      op += SP + SP + SP + SP + "remaining = elem_idx;\n";
      for (std::size_t d = 0; d < D; ++d) {
         std::string stride_val = "(" + outStrides[d].GetVal() + ")";
         op += SP + SP + SP + SP + "std::size_t const out_" + std::to_string(d)
               + " = remaining / " + stride_val + ";\n";
         op += SP + SP + SP + SP + "remaining -= out_" + std::to_string(d)
               + " * " + stride_val + ";\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "std::size_t chosen = 0;\n";
      for (std::size_t k = 0; k < Nin; ++k) {
         std::string end_k = "(" + prefix[k] + " + (" + fInputShapes[k][fAxis].GetVal() + "))";
         op += SP + SP + SP + SP + "chosen += static_cast<std::size_t>("
               + end_k + " <= out_" + std::to_string(fAxis) + ");\n";
      }
      op += "\n";

      op += SP + SP + SP + SP + "std::size_t const output_idx =\n";
      for (std::size_t d = 0; d < D; ++d) {
         op += SP + SP + SP + SP + SP + "out_" + std::to_string(d)
               + " * (" + outStrides[d].GetVal() + ")";
         op += (d + 1 < D) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "std::size_t const input_idx =\n";
      for (std::size_t k = 0; k < Nin; ++k) {
         op += SP + SP + SP + SP + SP + "(chosen == " + std::to_string(k) + "u) * (\n";
         for (std::size_t d = 0; d < D; ++d) {
               std::string coord = (d == static_cast<std::size_t>(fAxis))
                  ? ("(out_" + std::to_string(d) + " - (" + prefix[k] + "))")
                  : ("out_" + std::to_string(d));
               op += SP + SP + SP + SP + SP + SP + coord
                  + " * (" + inStrides[k][d].GetVal() + ")";
               op += (d + 1 < D) ? " +\n" : "\n";
         }
         op += SP + SP + SP + SP + SP + ")";
         op += (k + 1 < Nin) ? " +\n" : ";\n\n";
      }

      op += SP + SP + SP + SP + "output[output_idx] = inputs[chosen][input_idx];\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n";

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      if (fIsOutputConstant || fIsOutputParamShape) return "";
      opName = "op_" + opName;
      return SP + "ConcatKernel_" + opName + " concatKernel_" + opName + ";\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override {
      if (fIsOutputConstant || fIsOutputParamShape) return "";
      OpName = "op_" + OpName;
      if (fOutputShape.empty()) {
         throw std::runtime_error("SOFIE Operator Concat called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertDimShapeToLength(fOutputShape);
      out << "\n//------ CONCAT_GPU_ALPAKA\n";
      switch (fInputType){
         case ETensorType::FLOAT:
            out << SP << "std::array<const float *, " << fInputs.size() << "> input_ptrs_" << OpName << " = {"; break;
         case ETensorType::INT64:
            out << SP << "std::array<const int64_t *, " << fInputs.size() << "> input_ptrs_" << OpName << " = {"; break;
         default: 
            throw std::runtime_error("Data type for Concat operator is not yet supported.");
      }
      for(size_t i=0; i<fInputs.size(); ++i){
         if(i>0) out << ", ";
         out << "alpaka::getPtrNative(deviceBuf_" << fInputs[i] << ")";
      }
      out << "};\n";

      out << SP << "auto const elementsPerThread_"<<OpName<<" = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_"<<OpName<<" = Vec::all(Idx{"<< length << "});\n";
      out << SP << "auto const workDiv_" << OpName << " = sofie_workdiv(elementsPerGrid_" << OpName << ");\n";
      out << SP << "auto task_" << OpName << " = alpaka::createTaskKernel<Acc>(workDiv_" << OpName
         << ", concatKernel_" << OpName << ", input_ptrs_" << OpName << ", alpaka::getPtrNative(deviceBuf_" << fOutput << ")";
      for (auto &p : GetGPUDynParams())
         out << ", static_cast<std::size_t>(" << p << ")";
      out << ", static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::enqueue(queue, task_" << OpName << ");\n";
      return out.str();
   }

 };
 }//SOFIE


 #endif //SOFIE_ROPERATOR_CONCAT
