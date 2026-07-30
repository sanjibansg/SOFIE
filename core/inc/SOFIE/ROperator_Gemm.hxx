#ifndef SOFIE_ROPERATOR_GEMM
#define SOFIE_ROPERATOR_GEMM


#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>
#include <utility>


namespace SOFIE{


   template <typename T>
   class ROperator_Gemm final : public ROperator
   {

   private:
      bool fIsDynamic = false;
      bool fBroadcastBias = false;
      bool fCheckBiasShapeAtRuntime = false; // flag to identify the need to do a run time check of bias shape compatibility in case of dynamic shapes and uni-directional broadcasting

      float fAttrAlpha = 1.0;
      float fAttrBeta = 1.0;
      int_t fAttrTransA = 0;
      int_t fAttrTransB = 0;

      std::string fNA;
      std::string fNB;
      std::string fNC = "";
      std::string fNY;
      std::string fType;
      EActivationType fActivation;
      float fLeakyReluAlpha = 0.01f;   // used when fActivation == LEAKYRELU
      std::vector<Dim> fShapeA;
      std::vector<Dim> fShapeB;
      std::vector<size_t> fShapeC;
      std::vector<Dim> fDimShapeC;
      std::vector<Dim> fShapeY;
      RModel * fModel = nullptr;
      bool fForceFloatOutput = false;

   public:

      ROperator_Gemm(){}
      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameY, EActivationType activation=EActivationType::UNDEFINED, bool forceFloatOutput=false):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)), fForceFloatOutput(forceFloatOutput)
      {
         fActivation = activation;
         fType = "float";
         static_assert(std::is_same_v<T, float>,
                  "TMVA::SOFIE - Unsupported type parsing a Gemm operator");
         fInputTensorNames = { fNA, fNB };
         fOutputTensorNames = { fNY };
         fKind = OperatorKind::GEMM;
      }

      ROperator_Gemm(float alpha, float beta, int_t transA, int_t transB, std::string nameA, std::string nameB, std::string nameC, std::string nameY, EActivationType activation=EActivationType::UNDEFINED, bool forceFloatOutput=false):
         fAttrAlpha(alpha), fAttrBeta(beta), fAttrTransA(transA), fAttrTransB(transB), fNA(UTILITY::Clean_name(nameA)),
         fNB(UTILITY::Clean_name(nameB)), fNC(UTILITY::Clean_name(nameC)), fNY(UTILITY::Clean_name(nameY)), fActivation(activation),
         fForceFloatOutput(forceFloatOutput)
      {
         fActivation = activation;
         fType = "float";

         fInputTensorNames = {fNA, fNB, fNC};
         fOutputTensorNames = { fNY };
         fKind = OperatorKind::GEMM;
      }
      // Getters for further quantized GEMM lowering-elimination.
      float GetAlpha() const { return fAttrAlpha; }
      float GetBeta() const { return fAttrBeta; }
      int_t GetTransA() const { return fAttrTransA; }
      int_t GetTransB() const { return fAttrTransB; }
      bool HasBias() const { return !fNC.empty(); }
      const std::string &GetInputTensorName() const { return fNA; }
      const std::string &GetWeightTensorName() const { return fNB; }
      const std::string &GetBiasTensorName() const { return fNC; }
      const std::string &GetOutputTensorName() const { return fNY; }
      const std::vector<Dim> &GetInputShape() const { return fShapeA; }
      const std::vector<Dim> &GetWeightShape() const { return fShapeB; }
      const std::vector<Dim> &GetOutputShape() const { return fShapeY; }
      std::vector<std::string> GetStdLibs() override { return { std::string("cmath"), std::string("cstdint"), std::string("vector") }; }

      std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
         ETensorType out = input[0];
         if (fForceFloatOutput || out == ETensorType::FLOAT8E4M3FN || out == ETensorType::FLOAT8E4M3FNUZ ||
             out == ETensorType::FLOAT8E5M2 || out == ETensorType::FLOAT8E5M2FNUZ ||
             out == ETensorType::FLOAT8E8M0)
            out = ETensorType::FLOAT;
         return {out};
      }

      template <typename U>
      std::vector<U> DoShapeInference(const std::vector<std::vector<U>> & input){
         if (input.size() > 3) throw std::runtime_error("SOFIE Gemm Op Shape Inference only need 2 or 3 input tensor");
         // accept tensor with input dimensions > 2
         // example: A = (d1,d2,...,N1,N2)  B = (d1,d2,...,N2,N3)    --> Y = (d1,d2,..,N1,N3)
         for (auto& i: input){
            if (i.size() < 2){
               throw std::runtime_error("SOFIE Gemm Op Shape Inference only accept input tensor with >=2 dimensions");
            }
         }

         // GEMM output shape is determined by op(A) * op(B).  A third input C is
         // broadcast into that output and does not by itself define the result shape.
         // ioffset cannot be less than 2
         int ioffset = input[0].size()-2;  // in case of tensors with dim > 2

         std::vector<U> s_a(input[0].begin() + ioffset, input[0].begin() + ioffset + 2);
         std::vector<U> s_b(input[1].begin() + ioffset, input[1].begin() + ioffset + 2);
         // reverse in case of transpose
         if (fAttrTransA){
            std::reverse(s_a.begin(), s_a.end());
         }
         if (fAttrTransB){
            std::reverse(s_b.begin(), s_b.end());
         }
         std::vector<U> s_y;
         s_y.reserve(input[0].size());
         if (input[0].size() > 2 && input[1].size() == input[0].size()) {
            // in case of dim > 2 first dimensions are equal to the input ones not
            // equal to 1 (e.g. (1,2,3) * (2,3,4) -> (2,2,4))
            // here could probably use the Broadcasting function  UTILITY::MultidirectionalBroadcastShape
            for (size_t i = 0; i < input[0].size()-2; i++) {
               Dim valueA = input[0][i];
               Dim valueB = input[1][i];
               if (valueA.GetVal() != valueB.GetVal()) {
                  if (valueB.GetVal() == "1")
                     s_y.push_back(input[0][i]);
                  else if (valueA.GetVal() == "1")
                     s_y.push_back(input[1][i]);
                  else if (!valueA.isParam && !valueB.isParam)
                     throw std::runtime_error("SOFIE Gemm Op - invalid input shapes " + valueA.GetVal() + " and "
                        + valueB.GetVal());
                  else if (valueA.isParam && valueB.isParam){
                      // check which parameter is first in RModel list
                     auto & dimNames = fModel->GetDimShapeNames();
                     auto p1 = std::find(dimNames.begin(), dimNames.end(), valueA.param);
                     auto p2 = std::find(dimNames.begin(), dimNames.end(), valueB.param);
                     if (p1 < p2) s_y.push_back(input[0][i]);
                     else  s_y.push_back(input[1][i]);
                  }
                  else if (!valueA.isParam)
                     s_y.push_back(input[0][i]);
                  else if (!valueB.isParam)
                     s_y.push_back(input[1][i]);
                  else
                     throw std::runtime_error("SOFIE Gemm Op - invalid input shapes " + valueA.GetVal() + " and "
                        + valueB.GetVal());
               }
               else
                  s_y.push_back(input[0][i]);
            }
         }

         s_y.push_back(s_a[0]);
         s_y.push_back(s_b[1]);
         return s_y;
      }

      std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
         std::vector<std::vector<size_t>> ret;
         ret.push_back(DoShapeInference<size_t>(input));
         return ret;
      }
      std::vector<Dim> DynamicShapeInference(const std::vector<std::vector<Dim>> & input){
         return DoShapeInference<Dim>(input);
      }



      void Initialize(RModel& model) override {
         //TODO: propagate A or B as specified by ONNX standard
         fModel = &model;

         if ((model.CheckIfTensorAlreadyExist(fNA) == false) || (model.CheckIfTensorAlreadyExist(fNB) == false) ){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("SOFIE Gemm Op Input Tensor " + fNA + " or " + fNB + " is not found in model");
         }
         if (fNC != ""){
            if (model.CheckIfTensorAlreadyExist(fNC) == false){   //input must be a graph input, or already initialized intermediate tensor
               throw std::runtime_error("SOFIE Gemm Op Input Tensor " + fNC + " is not found in model");
            }
         }
         if (model.IsDynamicTensor(fNA) || model.IsDimInputTensor(fNA) ) {
            fShapeA = model.GetDynamicTensorShape(fNA);
            fIsDynamic = true;
         } else {
            auto shapeA_int = model.GetTensorShape(fNA);
            fShapeA = ConvertShapeToDim(shapeA_int);
         }
         // case A is of dim1 we prepend a 1 but we need to remove later
         bool prependOne = false;
         if (fShapeA.size() == 1) {
            fShapeA.insert(fShapeA.begin(), Dim(1));
            prependOne = true;
         }

         if (model.IsDynamicTensor(fNB) || model.IsDimInputTensor(fNB)) {
            fShapeB = model.GetDynamicTensorShape(fNB);
            fIsDynamic = true;
         }
         else {
            auto shapeB_int = model.GetTensorShape(fNB);
            fShapeB = ConvertShapeToDim(shapeB_int);
         }
         // case B is dim1 we append a 1 but we need to remove later
         bool appendOne = false;
         if (fShapeB.size() == 1) {
            fShapeB.insert(fShapeB.end(), Dim(1));
            appendOne = true;
         }
         // assume if not shape is 2 that extra values are 1.
         // implement also MatMul case where we stack matrices (see numpy.matmul)
         if (fShapeA.size() != fShapeB.size()) {
            // if different dimensions we prepend 1 values
            if (fShapeA.size() < fShapeB.size()) {
               fShapeA.insert(fShapeA.begin(), fShapeB.size()-fShapeA.size(), Dim(1));
            } else if (fShapeB.size() < fShapeA.size()) {
               fShapeB.insert(fShapeB.begin(), fShapeA.size()-fShapeB.size(), Dim(1));
            }
         }

         fShapeY = DynamicShapeInference({fShapeA, fShapeB});
         std::vector<size_t> shapeY = ConvertShapeToInt(fShapeY);

         // bias is normally not dynamic (not support it for time being)
         if (fNC != ""){
            if (model.IsDynamicTensor(fNC))
               fDimShapeC = model.GetDynamicTensorShape(fNC);
            else {
               fShapeC = model.GetTensorShape(fNC);
               fDimShapeC = ConvertShapeToDim(fShapeC);
            }
            // for dynamic outputs broadcasting is always needed
            bool broadcast_needed = false;
            if (fIsDynamic && shapeY.empty())
               broadcast_needed = true;
            else
               // consider broadcasting also if they have different length
               broadcast_needed = (fShapeC != shapeY);


            if (broadcast_needed) {
               fBroadcastBias = true;
               // check if broadcasting is compatible and note that prepend 1 to shapeC
               auto r = UTILITY::MultidirectionalBroadcastShape(fShapeY, fDimShapeC);
               // return flag must not have bit equal to 2 since this is a unidirectional broadcast of C->Y
               //
               if ((r.first & 2) == 2) {
                  throw std::runtime_error("SOFIE Gemm Op - bias tensor of shape " + ConvertDimShapeToString(fDimShapeC) + " cannot be uni-directional broadcasted to " + ConvertDimShapeToString(fShapeY));
               } else if (r.first  == 4) {
                  // we need to do a run time check of bias shape if it is compatible
                  fCheckBiasShapeAtRuntime = true;
               }
               fShapeC = ConvertShapeToInt(fDimShapeC);
            }
         }

         // remove appended or prepended value of 1 in Y
         if (prependOne) {
            if (fIsDynamic)
               fShapeY.erase(fShapeY.begin());
            else
               shapeY.erase(shapeY.begin());
         }
         if (appendOne) {
            if (fIsDynamic)
               fShapeY.erase(fShapeY.end()-1);
            else
               shapeY.erase(shapeY.end()-1);
         }

         // Use the same rule as TypeInference: a dense layer with an FP8 operand produces a
         // FLOAT value.
         const ETensorType outputType = TypeInference({model.GetTensorType(fNA)})[0];
         if (!fIsDynamic)
            model.AddIntermediateTensor(fNY, outputType, shapeY);
         else
            model.AddDynamicTensor(fNY, outputType, fShapeY);

         if (model.Verbose()){
            std::cout << "Gemm (or MatMul) " << " ---> " << fNY << " shape ";
            if (fIsDynamic)
               std::cout << ConvertDimShapeToString(fShapeY) << std::endl;
            else
               std::cout << ConvertShapeToString(shapeY) << std::endl;
         }

         model.AddNeededStdLib("algorithm");
      }

      std::string Generate(std::string opName) override {
         opName = "op_" + opName;

         // if (fShapeA.empty() || fShapeB.empty() || fShapeY.empty() || (fNC != "" && fShapeC.empty())) {
         //    throw std::runtime_error("SOFIE Gemm Op called to Generate without being initialized first");
         // }
         std::stringstream out;
         out << "\n//--------- Gemm " << opName << " " << ConvertDimShapeToString(fShapeA) << " * " << ConvertDimShapeToString(fShapeB)
             << " -> " << ConvertDimShapeToString(fShapeY) << "\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         int64_t dimC = fDimShapeC.size();
         if (dimA != dimB || dimA != dimY || (fBroadcastBias && dimC != dimY)) {
             std::cout << " shape A " << ConvertDimShapeToString(fShapeA)
                       << " shape B " << ConvertDimShapeToString(fShapeB)
                       << " shape C " << ConvertDimShapeToString(fDimShapeC)
                       << " shape Y " << ConvertDimShapeToString(fShapeY) << std::endl;
             throw std::runtime_error("SOFIE Gemm(MatMul) has invalid shape for inputs or output");
         }
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         // size of A: if (transposeA) is m*k else k*m
         // size of B  n*k
         std::vector<Dim> sY = {fShapeY[dimY-2], fShapeY[dimY-1]};
         // extra dimensions in case of stacked MatMul
         std::vector<Dim> sExtraY;
         for (int64_t i = 0; i < dimY-2; i++) {
            sExtraY.push_back(fShapeY[i]);
         }
         auto lengthGemm = ConvertDimShapeToLength(sY); // size of the Gemm operation
         auto lengthExtra_Y = ConvertDimShapeToLength(sExtraY); // extra length in case input tensors are of dim>2 (MatMul)
         std::string lengthExtra_C;
         std::vector<Dim> sExtraC;
         std::vector<Dim> sC;
         bool haveExtraC = false;
         if (dimC > 2) {
            sC = {fDimShapeC[dimC-2], fDimShapeC[dimC-1]};
            for (int64_t i = 0; i < dimC-2; i++) {
               sExtraC.push_back(fDimShapeC[i]);
            }
            lengthExtra_C = ConvertDimShapeToLength(sExtraC);
            if (lengthExtra_C != "1") haveExtraC = true;
         } else if (dimC > 0) {
            for (int64_t i = 0; i < dimC; i++) {
               sC.push_back(fDimShapeC[i]);
            }
         }

         // case bias is present
         if (!fNC.empty()){
             // when the 2 last dims of bias and Y are not compatible we need to perform a run time broadcast
            if (sC != sY) fBroadcastBias = true;
            if (!fBroadcastBias) {
               // add a check in case broadcasting was not needed or done outside of session
               // C should have smaller dimension of Y
               if (!fIsDynamic) {
                  if ((std::stoi(lengthGemm) != std::stoi(ConvertDimShapeToLength(sC))) ||
                      ( haveExtraC &&  std::stoi(lengthExtra_Y) != std::stoi(lengthExtra_C)))
                     throw std::runtime_error("SOFIE Gemm Op " + opName + " Bias tensor " + fNC + " has not correct size "
                            + ConvertShapeToString(fShapeC) + " output length " + lengthGemm);
               } else {
                  // add a dynamic check (C should not be a dynamic tensor)
                  out << SP << "assert(" << lengthGemm << " == " <<  ConvertDimShapeToLength(sC) << ");\n";
                  if (haveExtraC) out << SP << "assert(" << lengthExtra_Y << " == " <<  lengthExtra_C << ");\n";
               }
            }
         } else {
            fBroadcastBias = false;
            //in this case fAttrBeta needs to be equal to zero otherwise second time we run we will use
            // the previous result
            if (fAttrBeta != 0) {
               // some model don't have bias but Beta is not zero - force it to zero
               fAttrBeta = 0;
               std::cout << "WARNING: SOFIE Gemm Op " + opName + " Bias tensor is not present but beta value in Gemm is not zero - force it to zero\n";
            }
         }

         // include MatMul case where we stack the Gemm operations
         // exclude case where we have only 1's in the additional dims
         bool doStackMul = dimY > 2 && ( fIsDynamic  || std::stoi(lengthExtra_Y) > 1);
         // compute input offset for stack multiplications
         std::string lengthExtra_A;
         std::string lengthExtra_B;
         std::string increment_A;
         std::string increment_B;

         if (doStackMul) {
            std::vector<Dim> sA(fShapeA.begin(), fShapeA.begin()+dimA-2);
            std::vector<Dim> sB(fShapeB.begin(), fShapeB.begin()+dimB-2);
            std::vector<Dim> mA = {fShapeA[dimA-2], fShapeA[dimA-1]};
            std::vector<Dim> mB = {fShapeB[dimB-2], fShapeB[dimB-1]};
            lengthExtra_A = ConvertDimShapeToLength(sA);
            lengthExtra_B = ConvertDimShapeToLength(sB);
            // if A ( b, m, k) and B (b, k, n) these are the strides of A and B ( m*k for A and n*k for B )
            increment_A = ConvertDimShapeToLength(mA);
            increment_B = ConvertDimShapeToLength(mB);
         }
         bool extraA = (doStackMul && lengthExtra_A != "1");
         bool extraB = (doStackMul && lengthExtra_B != "1");
         bool extraC = (doStackMul && haveExtraC && !fBroadcastBias);
         // run time check for bias broadcasting
         std::string biasShapeType = opName + "_biasShapeType";
         if (fBroadcastBias && fCheckBiasShapeAtRuntime) {
            // create a flag according to bias shape:
            // = 1 for (1,Y2)
            // = 2 for (Y1,1)
            // = 3 for a scalar
            out << SP << "int " << biasShapeType << " = 0;\n";
            // case vector of columns
            if (sC[0].GetVal() != "1" && sC[1].GetVal() != sY[1].GetVal())
               out << SP << "if (" << sC[0] << " == 1 && " << sC[1] << " == " << sY[1] << ")\n";
            else if (sC[0].GetVal() == "1")
               out << SP << "if (" << sC[1] << " == " << sY[1] << ")\n";
            else if (sC[1].GetVal() == sY[1].GetVal())
               out << SP << "if (" << sC[0] << " == 1)\n";

            out << SP << SP << biasShapeType << " = 1;\n";

            // case vector of rows
            if (sC[1].GetVal() != "1" && sC[0].GetVal() != sY[0].GetVal())
               out << SP << "else if (" << sC[1] << " == 1 && " << sC[0] << " == " << sY[0] << ")\n";
            else if (sC[1].GetVal() == "1")
                out << SP << "else if (" << sC[0] << " == " << sY[0] << ")\n";
            else if (sC[0].GetVal() == sY[0].GetVal())
               out << SP << "else if (" << sC[1] << " == 1)\n";

            out << SP << SP << biasShapeType << " = 2;\n";

            // case scalar
            if (sC[0].GetVal() != "1" && sC[1].GetVal() != "1")
               out << SP << "else if (" << sC[0] << " == 1 && " << sC[1] << " == 1 )\n";
            else if (sC[0].GetVal() == "1")
               out << SP << "else if (" << sC[1] << " == 1)\n";
            else if (sC[1].GetVal() == "1")
               out << SP << "else if (" << sC[0] << " == 1)\n";
            out << SP << SP << biasShapeType << " = 3;\n";
            out << SP << "else\n";
            out << SP << SP << "throw std::runtime_error(\"SOFIE Gemm Op - bias tensor "
                                 << ConvertDimShapeToString(fDimShapeC) << " cannot be broadcasted to "
                                 << ConvertDimShapeToString(fShapeY) << "\");\n";
         }
         auto SP2 = SP;
         if (doStackMul) {
            out << SP << "size_t " << opName << "_y_offset = 0;\n"; // needed if we stack the gemm operations
            if (extraA)
               out << SP << "size_t " << opName << "_A_offset = 0;\n";
            if (extraB)
               out << SP << "size_t " << opName << "_B_offset = 0;\n";
            if (extraC)
               out << SP << "size_t " << opName << "_C_offset = 0;\n";
            out << SP << "for (size_t i = 0; i < " << lengthExtra_Y << "; i++){\n";
            SP2 += SP;
         }
         // do the bias broadcasting at run time by
         // initializing output Y vector with bias values
         if (fBroadcastBias) {

            fAttrBeta = 1.;

            // loop on first output dimension
            out << SP2 << "for (size_t j = 0; j < " << sY[0] << "; j++) { \n";
            out << SP2 << SP << "size_t y_index = ";
            if (doStackMul) // add offset in case of stack multiplications (not sure if bias is present in these cases)
               out <<  opName << "_y_offset + ";
            if (sY[1].GetVal() != "1")
               out << sY[1] << " * j;\n";
            else
               out << "j;\n";

            std::string prefix = SP2 + SP + "SOFIE::";
            std::string target = "tensor_" + fNY;
            if (sC.size() != 2) {
               throw std::runtime_error("SOFIE Gemm Op - invalid rank for bias tensor " + ConvertDimShapeToString(fDimShapeC) + ConvertDimShapeToString(sC));
            } if (sC[0].GetVal() == "1" && sC[1].GetVal() == sY[1].GetVal()) {
               out << prefix << "Copy(" << target << " + y_index, tensor_" << fNC << ", " << sY[1] << ");\n";
            } else if (sC[1].GetVal() == "1" && sC[0].GetVal() == sY[0].GetVal()) {
               out << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[j], " << sY[1] << ");\n";
            } else if (sC[0].GetVal() == "1" && sC[1].GetVal() == "1") {
               // scalar case
               out << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[0], " << sY[1] << ");\n";
            } else if (fCheckBiasShapeAtRuntime) {
               // in the generic dynamic case we check at run time that bias is compatible
               // we check that bias[0] = 1 or equal to SY[0] and that bias[1] = 1 or equal to SY[1]
               // tbd: this run-time check coul;d be moved outside the loop for better run time efficiency
               out << SP2 << SP << "if (" << biasShapeType << " == 1)\n";   // case vector of columns
               out << SP << prefix << "Copy(" << target << " + y_index, tensor_" << fNC << ", " << sY[1] << ");\n";
               out << SP2 << SP << "else if (" << biasShapeType << " == 2)\n";  // case vector of rows
               out << SP << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[j], " << sY[1] << ");\n";
               out << SP2 << SP << "else \n";  // scalar case
               out << SP << prefix << "Fill(" << target << " + y_index, tensor_" << fNC << "[0], " << sY[1] << ");\n";
            } else {
               throw std::runtime_error("SOFIE Gemm Op - invalid shape for bias tensor " + ConvertDimShapeToString(fDimShapeC));
            }

            out << SP2 << "}\n";
         }

         if (fType == "float"){

            out << SP2 << "SOFIE::Gemm_Call(" << "tensor_" << fNY;
             if (doStackMul) out << " + " << opName << "_y_offset";
            out <<   ", "
             << (fAttrTransB ? "true, " : "false, ")
             << (fAttrTransA ? "true, " : "false, ")
             << n << ", " << m << ", " << k << ", ";
            out << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ", tensor_" << fNB;
            if (extraB) out << " + " << opName << "_B_offset";
            out << ", tensor_" << fNA;
            if (extraA) out << " + " << opName << "_A_offset";
            out << ", " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrBeta << ",";
            // in the case of bias and no broadcasting needed - I need to add bias as an extra tensor in Gemm call
            if (!fNC.empty() && !fBroadcastBias) {
               out << "tensor_" << fNC;
               if (extraC) {
                  out << " + " << opName << "_C_offset";
               }
            } else {
               out << "nullptr";
            }
            out << ");\n";

         }

         if (doStackMul) {
            out << SP << SP <<  opName << "_y_offset += " << lengthGemm << ";\n";
            if (lengthExtra_A != "1")
               out << SP << SP << opName << "_A_offset += " << increment_A << ";\n";
            if (lengthExtra_B != "1")
               out << SP << SP << opName << "_B_offset += " << increment_B << ";\n";
            if (extraC)
               // increment_C is lengthGEmm
               out << SP << SP << opName << "_C_offset += " << lengthGemm << ";\n";
            out << SP << "}\n"; // end of loop on the stacked multiplication
         }

         // fuse activation with GEMM output (in-place on fNY)
         if (fActivation == EActivationType::RELU) {
               out << SP << "//--- applying RELU to output\n";
               std::string tnsr = "tensor_" + fNY;
               std::string reluSize = ConvertDimShapeToLength(fShapeY);
               out << SP << "SOFIE::Relu(" << tnsr << ", " << tnsr << ", " << reluSize << ");\n";
         } else if (fActivation == EActivationType::LEAKYRELU) {
               out << SP << "//--- applying LEAKYRELU to output (in-place)\n";
               std::string tnsr = "tensor_" + fNY;
               std::string reluSize = ConvertDimShapeToLength(fShapeY);
               out << SP << "{\n";
               out << SP << SP << "constexpr float lrelu_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fLeakyReluAlpha << "f;\n";
               out << SP << SP << "for (size_t _i = 0; _i < " << reluSize << "; ++_i)\n";
               out << SP << SP << SP << tnsr << "[_i] = " << tnsr << "[_i] >= 0.f ? " << tnsr << "[_i] : lrelu_alpha * " << tnsr << "[_i];\n";
               out << SP << "}\n";
         }

         return out.str();
      }

      std::string Generate_GPU_ALPAKA(std::string opName) override {
         opName = "op_" + opName;

         if (fShapeA.empty() || fShapeB.empty() || fShapeY.empty() || (fNC != "" && fDimShapeC.empty())) {
            throw std::runtime_error("SOFIE Gemm Op called to Generate without being initialized first");
         }
         std::stringstream out;
         out << "\n//--------- Gemm_GPU_ALPAKA\n";
         // Note: alpaka::wait(queue) intentionally removed here.
         // Operations are enqueued asynchronously on the Alpaka queue's CUDA
         // stream.  Synchronisation only happens once per inference at the
         // alpaka::wait(queue) call in _infer_impl's tail and at the
         // cudaDeviceSynchronize in the benchmark harness.  Adding a wait
         // before every GEMM stalls the CPU<->GPU pipeline and is the primary
         // cause of SOFIE being slower than ONNXRuntime.
         out << SP << "char " << opName << "_transA = " << (fAttrTransA ? "\'t\'" : "\'n\'") << ";\n";
         out << SP << "char " << opName << "_transB = " << (fAttrTransB ? "\'t\'" : "\'n\'") << ";\n";
         // need to consider case A and B have dim > 2 (for MatMul)
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         if (dimA != dimB || dimA != dimY) {
             throw std::runtime_error("SOFIE Gemm(MatMul) has invalid shape for inputs or output");
         }
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         std::vector<Dim> sY = {fShapeY[dimY-2], fShapeY[dimY-1]};
         // extra dimensions in case of stacked MatMul
         std::vector<Dim> sA;
         for (int64_t i = 0; i < dimY-2; i++) {
            sA.push_back(fShapeY[i]);
         }
         auto lengthGemm = ConvertDimShapeToLength(sY); // size of the Gemm operation
         auto lengthExtra = ConvertDimShapeToLength(sA); // extra length in case input tensors are of dim>2 (MatMul)

         out << SP << "int " << opName << "_m = " << m << ";\n";
         out << SP << "int " << opName << "_n = " << n << ";\n";
         out << SP << "int " << opName << "_k = " << k << ";\n";
         out << SP << "float " << opName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << fAttrAlpha << ";\n";
         
         // restricting to a 0 beta since BIAS is configured separately through sofieBLAS interface
         out << SP << "float " << opName << "_beta = 0;\n";

         // case bias is present
         if (!fNC.empty()){
            if (!fBroadcastBias) {
               // add a check in case broadcasting was not needed or done outside of session
               // C should have same size as Y
               if (!fIsDynamic) {
                  if (std::stoi(lengthGemm) != static_cast<int>(ConvertShapeToLength(fShapeC)))
                     throw std::runtime_error("SOFIE Gemm Op " + opName + " Bias tensor has not correct size "
                            + ConvertDimShapeToString(fDimShapeC) + " output length " + lengthGemm);
               } else {
                  // add a dynamic check (C should equal output size)
                  out << SP << "assert(" << lengthGemm << " == " <<  ConvertDimShapeToLength(fDimShapeC) << ");\n";
               }
            }
         } else {
            fBroadcastBias = false;
            //in this case fAttrBeta needs to be equal to zero otherwise second time we run we will use
            // the previous result
            if (fAttrBeta != 0) {
               // some model don't have bias but Beta is not zero - force it to zero
               fAttrBeta = 0;
               std::cout << "WARNING: SOFIE Gemm Op " + opName + " Bias tensor is not present but beta value in Gemm is not zero - force it to zero\n";
            }
         }

         // include MatMul case where we stack the Gemm operations
         // exclude case where we have only 1's in the additional dims
         bool doStackMul = dimY > 2 && ( fIsDynamic  || std::stoi(lengthExtra) > 1);

         // Compute per-iteration strides for each buffer when stacking.
         // m/n/k are std::string from Dim::GetVal(); stoi() is safe for static shapes.
         size_t strideA = 0, strideB = 0, strideY = 0, strideC = 0;
         // GPU optimisation flags (static shapes only):
         //   batchCollapseB  — strideB==0: B is the shared weight, so replace the N-iteration
         //                     loop with a single cuBLASLt GEMM whose batch dimension is
         //                     folded into the "n_sofie" parameter (n_sofie = m_onnx * N).
         //                     This turns 30 per-token GEMM launches into one kernel call.
         //   useSBatched     — both strides non-zero AND no bias: use cublasSgemmStridedBatched
         //                     so the GPU driver schedules all N GEMMs in one call.
         //                     (Bias epilogue is not available on the strided-batched path, so
         //                      this only applies to pure MatMul ops such as softmax(QK^T)·V.)
         bool batchCollapseB = false;
         bool useSBatched    = false;
         if (doStackMul && !fIsDynamic) {
            strideA = static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(k));
            // B is a shared weight (broadcast over the stacked/batch dimension) when all its
            // leading dims (beyond the 2 matrix dims) are 1.  In that case strideB must be 0
            // so every iteration reads from the same B slice — not i * n*k (which goes OOB).
            bool bLeadingDimsAllOne = true;
            for (int64_t i = 0; i < dimB - 2; i++) {
               if (fShapeB[i].dim != 1) { bLeadingDimsAllOne = false; break; }
            }
            strideB = bLeadingDimsAllOne ? 0
                                         : static_cast<size_t>(std::stoi(n)) * static_cast<size_t>(std::stoi(k));
            strideY = static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(n));
            strideC = !fNC.empty() ? static_cast<size_t>(std::stoi(lengthGemm)) : 0;

            batchCollapseB = (strideB == 0);
            useSBatched    = !batchCollapseB && fNC.empty();
         }

         // Emit the loop only for the serial fallback path (dynamic shapes, or static
         // shapes where both A and B vary per iteration AND a bias epilogue is needed).
         bool useSerialLoop = doStackMul && !batchCollapseB && !useSBatched;
         if (useSerialLoop || (doStackMul && fIsDynamic)) {
            out << SP << "size_t " << opName << "_yoffset = 0;\n";
            out << SP << "for (int i = 0; i < " << lengthExtra << "; i++){\n";
         }

         // Use getPtrNative() for all args so the raw-pointer overload is selected
         // regardless of whether each buffer is a BufXxx or ViewPlainPtr.
         // For the loop path, add per-iteration offsets; for the collapsed/batched
         // paths, use base pointers (the whole contiguous tensor is processed at once).
         std::string pA = "alpaka::getPtrNative(deviceBuf_" + fNA + ")";
         std::string pB = "alpaka::getPtrNative(deviceBuf_" + fNB + ")";
         std::string pY = "alpaka::getPtrNative(deviceBuf_" + fNY + ")";
         if (useSerialLoop && !fIsDynamic) {
            pA += " + i * " + std::to_string(strideA);
            if (strideB > 0) pB += " + i * " + std::to_string(strideB);
            // strideB == 0: B is a shared weight, pointer stays at base
            pY += " + i * " + std::to_string(strideY);
         }

         if (useSBatched) {
            // ----------------------------------------------------------------
            // gemmStridedBatched: both A and B vary per batch (e.g. per attention
            // head), and there is no bias.  Uses cublasSgemmStridedBatched via
            // the legacy cuBLAS handle so all N GEMMs are issued in one driver call.
            //
            // sofieBLAS convention (column-major transpose trick):
            //   transa_sofie = transB_onnx,  transb_sofie = transA_onnx
            //   m_sofie      = n_onnx,        n_sofie      = m_onnx
            //   A_sofie      = fNB,           B_sofie      = fNA
            //   lda = m_sofie  (leading dim of A when transA_sofie='n')
            //   ldb = k        (leading dim of B when transB_sofie='n')
            //   ldc = m_sofie  (leading dim of C)
            // ----------------------------------------------------------------
            size_t m_sofie    = static_cast<size_t>(std::stoi(n));   // ONNX n
            size_t n_sofie    = static_cast<size_t>(std::stoi(m));   // ONNX m
            size_t k_val      = static_cast<size_t>(std::stoi(k));
            size_t lda        = m_sofie;             // transA_sofie='n'
            size_t ldb        = k_val;               // transB_sofie='n'
            size_t ldc        = m_sofie;
            size_t sA         = m_sofie * k_val;     // stride per batch for fNB
            size_t sB         = k_val  * n_sofie;    // stride per batch for fNA (= strideA_onnx)
            size_t sC         = m_sofie * n_sofie;   // stride per batch for fNY (= strideY)
            size_t batchCount = static_cast<size_t>(std::stoi(lengthExtra));
            out << SP << "blas.gemmStridedBatched("
                << opName << "_transB, " << opName << "_transA, "
                << m_sofie << ", " << n_sofie << ", " << k_val << ", "
                << opName << "_alpha, "
                << "alpaka::getPtrNative(deviceBuf_" << fNB << "), "
                << lda << ", " << sA << ", "
                << "alpaka::getPtrNative(deviceBuf_" << fNA << "), "
                << ldb << ", " << sB << ", "
                << opName << "_beta, "
                << "alpaka::getPtrNative(deviceBuf_" << fNY << "), "
                << ldc << ", " << sC << ", "
                << batchCount << ");\n";
         } else if (!fNC.empty()) {
            // ----------------------------------------------------------------
            // GEMM with bias:  Y = alpha * op(A) * op(B) + bias
            // cuBLAS is column-major so we swap A↔B and transA↔transB
            // (row-major C=A*B  ↔  col-major C^T = B^T * A^T).
            // The epilogue fuses the bias-add (and optional ReLU/GELU) in the
            // same kernel, avoiding a separate element-wise pass.
            //
            // For batch-collapse (batchCollapseB), use m*batchCount so that all
            // tokens are processed in a single cuBLASLt kernel launch instead of N.
            // The bias vector is broadcast across all columns by the epilogue.
            // ----------------------------------------------------------------
            std::string call_m = batchCollapseB
               ? std::to_string(static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(lengthExtra)))
               : (opName + "_m");

            std::string pC = "alpaka::getPtrNative(deviceBuf_" + fNC + ")";
            if (useSerialLoop && !fIsDynamic) {
               if (!fBroadcastBias && strideC > 0)
                  pC += " + i * " + std::to_string(strideC);
            }
            if (fActivation == EActivationType::RELU) {
               out << SP << "blas.gemmrelu("
                   << opName << "_transB, " << opName << "_transA, "
                   << opName << "_n, "      << call_m << ", "
                   << opName << "_k, "      << opName << "_alpha, "
                   << pB << ", " << pA << ", "
                   << opName << "_beta, " << pC << ", " << pY << ");\n";
            } else {
               out << SP << "blas.gemm("
                   << opName << "_transB, " << opName << "_transA, "
                   << opName << "_n, "      << call_m << ", "
                   << opName << "_k, "      << opName << "_alpha, "
                   << pB << ", " << pA << ", "
                   << opName << "_beta, " << pC << ", " << pY << ");\n";
            }
         } else {
            // ----------------------------------------------------------------
            // Pure MatMul (no bias):  Y = alpha * op(A) * op(B)
            // This covers:
            //   • Scaled Dot-Product Attention:  softmax(QK^T/√d) @ V
            //   • Any other no-bias matrix multiplication
            // Previously this branch emitted nothing (empty loop body), which
            // caused the attention output to be silently uninitialized.
            // For batch-collapse, use m*batchCount for the same reason as above.
            // ----------------------------------------------------------------
            std::string call_m = batchCollapseB
               ? std::to_string(static_cast<size_t>(std::stoi(m)) * static_cast<size_t>(std::stoi(lengthExtra)))
               : (opName + "_m");

            out << SP << "blas.matmul("
                << opName << "_transB, " << opName << "_transA, "
                << opName << "_n, "      << call_m << ", "
                << opName << "_k, "      << opName << "_alpha, "
                << pB << ", " << pA << ", "
                << opName << "_beta, "  << pY << ");\n";
         }

         if (useSerialLoop || (doStackMul && fIsDynamic)) {
            out << SP << "}\n"; // end of loop on the stacked multiplication
         }

         // GEMM+LeakyReLU fusion (GPU): cuBLASLt has no native LeakyReLU epilogue,
         // so we emit a cheap in-place ALPAKA kernel immediately after the GEMM.
         // This avoids allocating a separate intermediate buffer and saves one
         // GPU kernel launch compared to a standalone LeakyReLU operator.
         if (fActivation == EActivationType::LEAKYRELU) {
            std::string numElem = ConvertDimShapeToLength(fShapeY);
            out << SP << "//--- GEMM+LeakyReLU in-place fusion\n";
            out << SP << "{\n";
            out << SP << SP << "constexpr float " << opName << "_lrelu_alpha = "
                << std::setprecision(std::numeric_limits<float>::max_digits10)
                << fLeakyReluAlpha << "f;\n";
            out << SP << SP << "auto const elementsPerThread_lrelu_" << opName
                << " = Vec::all(static_cast<Idx>(1));\n";
            out << SP << SP << "auto const elementsPerGrid_lrelu_" << opName
                << " = Vec::all(Idx{" << numElem << "});\n";
            out << SP << SP << "auto const workDiv_lrelu_" << opName
                << " = sofie_workdiv(elementsPerGrid_lrelu_" << opName << ");\n";
            // In-place: input and output pointer are the same device buffer.
            out << SP << SP << "auto task_lrelu_" << opName
                << " = alpaka::createTaskKernel<Acc>(workDiv_lrelu_" << opName
                << ", leakyReluKernel"
                << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
                << ", alpaka::getPtrNative(deviceBuf_" << fNY << ")"
                << ", static_cast<Idx>(" << numElem << ")"
                << ", static_cast<float>(" << opName << "_lrelu_alpha));\n";
            out << SP << SP << "alpaka::enqueue(queue, task_lrelu_" << opName << ");\n";
            out << SP << "}\n";
         }

         return out.str();
      }

      std::vector<std::string> GetBlasRoutines() override { return { std::string("Gemm"), std::string("Gemv") }; }
      std::string GetFusableOutputTensorName() override {
         return fNY;
      }

      void UpdateFusableTensorName(std::string fusable_tensor_name, const std::function<void(const std::string&)>& removal_func){
         removal_func(fNY);
         fNY = fusable_tensor_name;
         fOutputTensorNames[0] = fNY;
      }

      // --- Activation fusion accessors (used by FuseGemmActivations_GPU) ---
      EActivationType GetActivationType() const { return fActivation; }
      /// Set fused activation.  alpha is only meaningful for LEAKYRELU.
      void SetActivation(EActivationType act, float alpha = 0.f) {
         fActivation      = act;
         fLeakyReluAlpha  = alpha;
      }

      std::string GetBlasConfig(){
         int64_t dimA = fShapeA.size();
         int64_t dimB = fShapeB.size();
         int64_t dimY = fShapeY.size();
         auto m = (fAttrTransA ? fShapeA[dimA-1].GetVal() : fShapeA[dimA-2].GetVal());
         auto n = (fAttrTransB ? fShapeB[dimB-2].GetVal() : fShapeB[dimB-1].GetVal());
         auto k = (fAttrTransA ? fShapeA[dimA-2].GetVal() : fShapeA[dimA-1].GetVal());
         auto lda = (fAttrTransA ? m : k);
         auto ldb = (fAttrTransB ? k : n);
         auto ldc = n;
         std::string transFlags = std::string(fAttrTransB ? "'t'" : "'n'") + ", " + (fAttrTransA ? "'t'" : "'n'");

         // For stacked (batched) GEMMs on static shapes, return the layout that
         // matches the actual call emitted by Generate_GPU_ALPAKA:
         //   - batch-collapse (strideB==0): single GEMM with n_sofie = m_onnx * batchCount
         //                                  → register the batched layout
         //   - gemmStridedBatched (both strides non-zero, no bias): uses legacy cuBLAS,
         //                                  no cuBLASLt layout needed → return ""
         if (dimY > 2 && !fIsDynamic) {
            std::vector<Dim> sExtra;
            for (int64_t i = 0; i < dimY - 2; i++) sExtra.push_back(fShapeY[i]);
            auto lengthExtra = ConvertDimShapeToLength(sExtra);
            if (std::stoi(lengthExtra) > 1) {
               bool bLeadingDimsAllOne = true;
               for (int64_t i = 0; i < dimB - 2; i++) {
                  if (fShapeB[i].dim != 1) { bLeadingDimsAllOne = false; break; }
               }
               if (bLeadingDimsAllOne) {
                  // batch-collapse: register layout for the full-batch GEMM
                  auto m_batched = std::to_string(std::stoi(m) * std::stoi(lengthExtra));
                  return n+", "+m_batched+", "+k+", "+ldb+", "+lda+", "+ldc+", "+transFlags;
               } else if (fNC.empty()) {
                  // gemmStridedBatched: legacy cuBLAS, no cuBLASLt layout needed
                  return "";
               }
               // else: serial loop with bias — fall through to per-iteration layout
            }
         }

         return n+", "+m+", "+k+", "+ldb+", "+lda+", "+ldc+", "+transFlags;
      }
   };


}//SOFIE

#endif //SOFIE_ROPERATOR_GEMM
