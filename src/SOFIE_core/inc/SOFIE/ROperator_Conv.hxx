#ifndef SOFIE_SOFIE_ROPERATOR_CONV
#define SOFIE_SOFIE_ROPERATOR_CONV

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>


namespace SOFIE {

template<typename T>
class ROperator_Conv final : public ROperator
{
private:
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   size_t fAttrGroup;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNW;
   std::string fNB;
   std::string fNB2; // bias tensor name after broadcasting
   std::string fNY;

   std::string convK;
   std::string imcol;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::string fType;

   size_t fDim;   // dimension of the convolution
   size_t gemm_n, gemm_m, gemm_k; // dimensions of the equivalent gemm operation after im2col transformation


public:

   ROperator_Conv() {}

   ROperator_Conv(std::string autopad, std::vector<size_t> dilations,
      size_t group, std::vector<size_t> kernelShape, std::vector<size_t> pads,
      std::vector<size_t> strides, std::string nameX, std::string nameW,
      std::string nameB, std::string nameY):
      fAttrAutopad(autopad), fAttrDilations(dilations), fAttrGroup(group), fAttrKernelShape(kernelShape),
      fAttrPads(pads), fAttrStrides(strides),
      fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
      fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Conv operator");
      }
      fInputTensorNames = { fNX, fNB };
      fOutputTensorNames = { fNY };
      fKind = OperatorKind::CONV;
   }

   ROperator_Conv(std::string autopad, std::vector<size_t> dilations,
      size_t group, std::vector<size_t> kernelShape, std::vector<size_t> pads,
      std::vector<size_t> strides, std::string nameX, std::string nameW,
      std::string nameY):
      fAttrAutopad(autopad), fAttrDilations(dilations), fAttrGroup(group), fAttrKernelShape(kernelShape),
      fAttrPads(pads), fAttrStrides(strides),
      fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Conv operator");
      }
      fInputTensorNames = { fNX };
      fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      ETensorType out = input[0];
      return {out};
   }

   // function returning output shape given input
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      // shape of convolution input has to be (according to ONNX): N x C x H x W
      // Where N : batch size, C : input  channels, H : input height, W : input width

      if (input.size() > 3 ) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op Shape inference need 2 or 3 input tensors");
      }
      for(size_t i = 0; i < input.size(); i++) {
         if (input[i].size() -2 != fDim) {
            throw
               std::runtime_error("TMVA SOFIE Conv Op Shape inference - invalid inputs ");
         }
      }

      if (fAttrGroup == 0) {
         fAttrGroup = input[0][1] / input[1][1];
      }

      // kernel shape
      size_t k1 = ((fAttrKernelShape.empty())? input[1][2] : fAttrKernelShape[0]);
      size_t k2 = (fDim > 1) ? ((fAttrKernelShape.empty()) ? input[1][3] : fAttrKernelShape[1]) : 1;
      size_t k3 = (fDim > 2) ? ((fAttrKernelShape.empty()) ? input[1][4] : fAttrKernelShape[2]) : 1;


      size_t i1 = (fDim > 1) ? ((fDim > 2) ? 3 : 2) : 1;
      size_t i2 = (fDim > 2) ? 4 : 3;
      size_t i3 = 5;

      if (fAttrDilations.empty()) {
         fAttrDilations = {1, 1, 1};
      }
      fAttrDilations.resize(3);
      if (fDim < 3) {
         fAttrDilations.resize(3, 1);
      }
      // Shape of the kernel
      fAttrKernelShape = {k1 + (fAttrDilations[0] - 1) * (k1 - 1),
                          k2 + (fAttrDilations[1] - 1) * (k2 - 1),
                          k3 + (fAttrDilations[2] - 1) * (k3 - 1)};

      if (fAttrAutopad == "NOTSET") {
         if (fAttrPads.empty()) {
            fAttrPads = {1, 1, 1, 1, 1, 1};
         }
      } else if (fAttrAutopad == "SAME_UPPER" || fAttrAutopad == "SAME_LOWER") {
         if (fDim == 1)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[0] / 2};
         else if (fDim == 2)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2};
         else if (fDim == 3)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[2] / 2,
                         fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[2] / 2};
         // add extra padding at beginning or end (depending if SAME_UPPER or SAME_LOWER)
         // need to check this!
         if (fAttrKernelShape[0] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[0]++ : fAttrPads[i1]++;
         }
         if (fDim > 1 && fAttrKernelShape[1] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[1]++ : fAttrPads[i2]++;
         }
         if (fDim > 2 && fAttrKernelShape[2] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[2]++ : fAttrPads[i3]++;
         }
      } else if (fAttrAutopad != "VALID") {
         throw
            std::runtime_error("TMVA SOFIE Conv Op invalid fAutopad");
      }
      // to be sure pad is vector of size 6
      if (fDim < 3) fAttrPads.resize(6, 0);

      if (fAttrStrides.empty()) {
         fAttrStrides = {1, 1, 1};
      }
      if (fDim < 3)
         fAttrStrides.resize(3, 1);


      size_t input1 = input[0][2];
      size_t input2 = (fDim > 1) ? input[0][3] : 1;
      size_t input3 = (fDim > 2) ? input[0][4] : 1;

      size_t pad1 = fAttrPads[0] + fAttrPads[i1];
      size_t output1 = (input1 + pad1 - fAttrKernelShape[0]) / fAttrStrides[0] + 1;

      size_t batch_size = input[0][0];        // first element in input tensor
      size_t output_channels = input[1][0];   // first element in weight tensor

      std::vector<std::vector<size_t>> ret({{ batch_size, output_channels, output1 }});

      if (fDim == 1)
         return ret;

      size_t pad2 = fAttrPads[1] + fAttrPads[i2];
      size_t output2 = (input2 + pad2 - fAttrKernelShape[1]) / fAttrStrides[1] + 1;
      // output is N x M x OH x OW
      ret[0].push_back(output2);
      if (fDim == 2)
         return ret;

      size_t pad3 = fAttrPads[2] + fAttrPads[i3];
      size_t output3 = (input3 + pad3 - fAttrKernelShape[2] ) / fAttrStrides[2] + 1;

      // output is N x M x OH x OW x OD
      ret[0].push_back(output3);
      return ret;
   }

   void Initialize(RModel& model) override {
      fUseSession = model.UseSession();
      if (!model.CheckIfTensorAlreadyExist(fNX))
         throw std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNX + " is not found in model");

      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() < 3 || fShapeX.size() > 5)
         throw std::runtime_error("TMVA SOFIE Conv Op input data tensor" + fNX + " is not of 3,4 or 5 dimensions");

      fDim = fShapeX.size() - 2;

      if (!model.CheckIfTensorAlreadyExist(fNW))
         throw std::runtime_error("TMVA SOFIE Conv op Input weight Tensor " + fNW + " is not found in model");

      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() < 3 || fShapeW.size() > 5)
         throw std::runtime_error("TMVA SOFIE Conv Op input weight tensor" + fNW + " is not of 3,4 or 5 dimensions");

      fShapeY = ShapeInference({fShapeX, fShapeW})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      if (fNB != "") {
         if (!model.CheckIfTensorAlreadyExist(fNB))
               throw std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNB + " is not found in model");

         fShapeB = model.GetTensorShape(fNB);
         std::vector<size_t> targetShape(fShapeY.begin() + 1, fShapeY.end());
         bool broadcast_needed = !UTILITY::AreSameShape(fShapeB, targetShape);
         if (broadcast_needed) {
               auto original_data = model.GetInitializedTensorData(fNB);
               if (fShapeB.size() < 1)
                  throw std::runtime_error("TMVA SOFIE Conv op: Bias Tensor has empty shape");
               if (fShapeB[0] != fShapeY[1])
                  throw std::runtime_error("TMVA SOFIE Conv op: Bias Tensor has wrong shape: " +
                                          ConvertShapeToString(fShapeB));
               if (fType != "float")
                  throw std::runtime_error("TMVA SOFIE Conv op: Broadcasting for non-float type tensors is not supported");
               if (!fUseSession) {
                  std::vector<size_t> shape(fDim + 1, 1);
                  shape[0] = fShapeB[0];
                  std::shared_ptr<void> new_data_ptr(
                     UTILITY::UnidirectionalBroadcast<float>(static_cast<float *>(original_data.get()), shape, targetShape),
                     std::default_delete<float[]>());
                  model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), targetShape, new_data_ptr);
                  fShapeB = model.GetTensorShape(fNB);
                  fNB2 = fNB;
               } else {
                  fNB2 = fNB + "bcast";
                  model.AddIntermediateTensor(fNB2, model.GetTensorType(fNB), targetShape);
               }
         }
      }

      // Compute kernel size once — product of all spatial kernel dimensions
      size_t kernelSize = fAttrKernelShape[0];
      for (size_t i = 1; i < fDim; i++)
         kernelSize *= fAttrKernelShape[i];

      // Spatial output size: OH * OW (* OD for 3D)
      size_t spatialSize = fShapeY[2];
      for (size_t i = 1; i < fDim; i++)
         spatialSize *= fShapeY[2 + i];

      std::vector<size_t> shape1 = {fShapeW[0], fShapeW[1], kernelSize};
      std::vector<size_t> shape2 = {fShapeW[1], kernelSize, spatialSize};
      model.AddIntermediateTensor(fNX + "_f",    ConvertStringToType(fType), shape1);
      model.AddIntermediateTensor(fNX + "_xcol", ConvertStringToType(fType), shape2);
      convK = fNX + "_f";
      imcol = fNX + "_xcol";
      fOutputTensorNames.emplace_back(convK);
      fOutputTensorNames.emplace_back(imcol);

      // GEMM dimensions — set once here, reused in Generate() and Generate_GPU_ALPAKA()
      // gemm_n = output channels (total, not per group — group case divides at launch time)
      // gemm_m = spatial output size (OH * OW * OD)
      // gemm_k = input channels per group * kernel spatial size
      gemm_n = fShapeW[0];                   // total output channels
      gemm_m = spatialSize;                  // OH * OW (* OD)
      gemm_k = fShapeW[1] * kernelSize;      // IC_per_group * KH * KW (* KD)
      if(fAttrGroup > 1) {
         gemm_n /= fAttrGroup;
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      // Generate initialization code for broadcasting of bias tensor
      if (!fNB2.empty()) {
         // include a separate scope to avoid defining unique operator temp variables
         std::vector<size_t> shape(fDim + 1, 1);
         shape[0] = fShapeB[0];
         std::vector<size_t> targetShape(fShapeY.begin() + 1, fShapeY.end());
         out << SP << "{\n";
         out << SP << SP << "float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_"
             << fNB << ", " << ConvertShapeToString(shape) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << ConvertShapeToLength(targetShape) << ", tensor_" << fNB2 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeW.empty() || (fNB != "" && fShapeB.empty()) || fShapeY.empty()) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");
      }

      std::stringstream out;
      size_t bsize = fShapeX[0];
      size_t kDepth = (fDim > 2) ?  fShapeW[2] : 1;  // kernel depth
      size_t kHeight = (fDim > 1) ? fShapeW[fDim] : 1;  // kernel height
      size_t kWidth = fShapeW[fDim+1]; // kernel width
      size_t iDepth = (fDim > 2) ?  fShapeX[2] : 1;  // input depth
      size_t iHeight = (fDim > 1) ? fShapeX[fDim] : 1; // input height
      size_t iWidth = fShapeX[fDim+1]; // input width
      size_t oDepth = (fDim > 2) ? fShapeY[2] : 1; // output depth
      size_t oHeight = (fDim > 1) ? fShapeY[fDim] : 1;  // ouput height
      size_t oWidth = fShapeY[fDim+1]; // output width

      out << "\n//----  operator Conv " << OpName << "\n";

      // vectorize the (dilated)convolution kernels into a matrix
      // no need to transpose the matrix
      // to fix for 1d and 3d

      size_t id = (fDim > 2) ? fDim-3 : 2;
      size_t ih = (fDim > 1) ? fDim-2 : 1;
      size_t iw = fDim-1;

      size_t wstrideDil = fAttrDilations[iw];
      size_t hstride = kWidth;
      size_t hstrideDil = fAttrDilations[ih] * fAttrKernelShape[iw];  // stride dilated in the height
      size_t dstride = kHeight * kWidth;
      size_t dstrideDil = fAttrDilations[id] * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t icstride = kHeight * kWidth * kDepth;
      size_t icstrideDil = fAttrKernelShape[id] * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t ocstride = fShapeW[1] * icstride;
      size_t ocstrideDil = fShapeW[1] * icstrideDil;

      out << SP << "for (std::size_t oc = 0; oc < " << fShapeW[0] << "; oc++) {\n";
      out << SP << SP << "for (std::size_t ic = 0; ic < " << fShapeW[1] << "; ic++) {\n";
      if (fDim > 2)
         out << SP << SP << SP << "for (std::size_t kd = 0; kd < " << kDepth << "; kd++) {\n";
      if (fDim > 1)
         out << SP << SP << SP << "for (std::size_t kh = 0; kh < " << kHeight << "; kh++) {\n";
      out << SP << SP << SP << SP << "for (std::size_t kw = 0; kw < " << kWidth << "; kw++) {\n";

      out << SP << SP << SP << SP << SP << "tensor_" <<fNX <<  "_f[oc * "
          << ocstrideDil << " + ic * " << icstrideDil;
      if (fDim > 2) out << " + kd * " << dstrideDil;
      if (fDim > 1) out << " + kh * " << hstrideDil;
      out << " + kw * " << wstrideDil  << "  ] = tensor_" << fNW << "[oc * " << ocstride << " + ic * " << icstride;
      if (fDim > 2) out << " + kd * " << dstride;
      if (fDim > 1) out << " + kh * " << hstride;
      out  << " + kw ];\n";

      out << SP << SP << SP << SP << "}\n";
      if (fDim > 1) out << SP << SP << SP << "}\n";
      if (fDim > 2) out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";

      //out << SP << "char " << OpName << "_transA = 'T';\n";
      out << SP << "char " << OpName << "_transA = 'N';\n";
      out << SP << "char " << OpName << "_transB = 'N';\n";
      out << SP << "int " << OpName << "_m = " << oHeight * oWidth * oDepth << ";\n"; // output h*w
      assert(fShapeY[1] == fShapeW[0]);
      assert(fShapeW[1] == fShapeX[1] / fAttrGroup);
      out << SP << "int " << OpName << "_n = " << fShapeW[0] << ";\n"; // output channels
      out << SP << "int " << OpName << "_k = " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2] << ";\n";
      out << SP << "float " << OpName << "_alpha = 1.0;\n";
      out << SP << "float " << OpName << "_beta = 0.0;\n";


      // Loop on batch size
      out << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";

      // IM2COL: Unroll the input tensor
      // order input data as  (e.g. kernel 2x2)  and (xa,ya) is channel 1 and (xb,yb) is channel 2
      //   (xa1,..,xak,ya1,..yak)(xb1,...,xbk,yb1,..,ybk)
      //   (xa2,...xak+1,ya1,...yak)(......)
      // trick for speed is using caffe im2col and output a matrix which contains filtered values as rows.
      // By doing this one has consecutive memory reads and writes
      // Resulting matrix op_xcol is (input channels * filter_h * filter_w , output_h * output_w)
      if (fDim ==1) {
         if (fAttrPads[0] != fAttrPads[1] ) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding "
                      << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[1]) / 2;
         }
         fAttrPads[1] = 0;
         fAttrStrides[1] = 1;
      }
      if (fDim == 2) {
         if (fAttrPads[0] != fAttrPads[2] || fAttrPads[1] != fAttrPads[3]) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding " << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[2]) / 2;
            fAttrPads[1] = (fAttrPads[1] + fAttrPads[3]) / 2;
         }
      }
      if (fDim == 3) {
         if (fAttrPads[0] != fAttrPads[3] || fAttrPads[1] != fAttrPads[4] || fAttrPads[2] != fAttrPads[5]) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding " << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[3]) / 2;
            fAttrPads[1] = (fAttrPads[1] + fAttrPads[4]) / 2;
            fAttrPads[2] = (fAttrPads[2] + fAttrPads[5]) / 2;
         }
      }
      out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << ";\n";

      if (fAttrGroup == 1) {
         out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iHeight * iWidth << ";\n";
         // when using im2col - resulting matrix is transposed, the dimension is (input_c * filter_h * filter_y,  output_h *
         // output_w)
         if (fDim < 3) {
            out << SP << SP << "SOFIE::UTILITY::Im2col<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                //  dilation_w,
                //
                << fShapeW[1] << "," << iHeight << "," << iWidth << ",";
            if (fDim == 1)
               out << "1, " << fAttrKernelShape[0] << ",0," << fAttrPads[0] << ",1," << fAttrStrides[0] << ",1,"
                   << fAttrDilations[0];
            else // dim ==2
               out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[1]
                   << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrDilations[0] << ","
                   << fAttrDilations[1];
            out << "," << "tensor_" <<fNX << "_xcol);\n\n ";
         } else {
            // 3d im2col
            out << SP << SP << "SOFIE::UTILITY::Im2col_3d<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, d, h, w, k_d, k_h, k_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                //  dilation_d, dilation_h, dilation_w,
                //
                << fShapeW[1] << "," << iDepth << "," << iHeight << "," << iWidth << ","
                << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << ","
                << fAttrPads[0] << "," << fAttrPads[1] << "," << fAttrPads[2] << ","
                << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2] << ","
                << fAttrDilations[0] << "," << fAttrDilations[1] << "," << fAttrDilations[2] << ","
                << "tensor_" << fNX << "_xcol);\n\n ";
         }
         // BLAS
         out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << "tensor_" << fNX << "_xcol, &" << OpName
             << "_m,\n"; // use m if op_xcol is not transpose , otherwise k
         out << SP << SP << SP << "tensor_" << fNX << "_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << " + out_offset, &" << OpName << "_m);\n";
      } else {
         // case of group convolution
         // Unroll (IM2COL) the input tensor- make loop on groups and repeat operations (IM2COL + GEMM for each
         // group)
         // out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << ";\n";
         out << SP << SP << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
         out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iDepth * iHeight * iWidth << " + g * "
             << fShapeW[1] * iDepth * iHeight * iWidth << ";\n ";
         out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << " + g * "
             << fShapeW[0] * oDepth * oHeight * oWidth / fAttrGroup << ";\n ";

         if (fDim < 3) {
            out << SP << SP << "SOFIE::UTILITY::Im2col<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                //  dilation_w,
                //
                << fShapeW[1] << "," << iHeight << "," << iWidth << ",";
            if (fDim == 1)
               out << "1, " << fAttrKernelShape[0] << ",0," << fAttrPads[0] << ",1," << fAttrStrides[0] << ",1,"
                   << fAttrDilations[0];
            else // dim ==2
               out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[1]
                   << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrDilations[0] << ","
                   << fAttrDilations[1];
            out << ", tensor_" << fNX << "_xcol);\n\n ";
         } else {
            // 3d im2col
            out << SP << SP << "SOFIE::UTILITY::Im2col_3d<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, d, h, w, k_d, k_h, k_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                //  dilation_d, dilation_h, dilation_w,
                //
                << fShapeW[1] << "," << iDepth << "," << iHeight << "," << iWidth << "," << fAttrKernelShape[0] << ","
                << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << "," << fAttrPads[0] << "," << fAttrPads[1]
                << "," << fAttrPads[2] << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2]
                << "," << fAttrDilations[0] << "," << fAttrDilations[1] << "," << fAttrDilations[2] << ",tensor_" << fNX
                << "_xcol);\n\n ";
         }

         // BLAS
         // n must be divided by the number of groups
         out << SP << SP << SP << OpName << "_n = " << fShapeW[0] / fAttrGroup << ";\n";
         // offset g must be  g * k * n
         out << SP << SP << SP << "size_t offset_f = g * "
             << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2] / fAttrGroup
             << ";\n";
         out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, tensor_" << fNX << "_xcol, &" << OpName
             << "_m,\n"; // use m if op_xcol is not transpose , otherwise k
         out << SP << SP << SP << "tensor_" << fNX << "_f + offset_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << " + out_offset"
             << ", &" << OpName << "_m);\n";

         out << SP << SP << "}\n"; // end of group loop
      }

      if (fNB2 != "") {
         out << SP << "int " << OpName << "_size = " << fShapeY[1] * oDepth * oHeight * oWidth << ";\n";
         out << SP << "float " << OpName << "_gamma = 1.0;\n";
         out << SP << "int " << OpName << "_incx = 1;\n";
         out << SP << "int " << OpName << "_incy = 1;\n";

         out << SP << "BLAS::saxpy_(&" << OpName << "_size, &" << OpName << "_gamma, tensor_" << fNB2 << ", &"
             << OpName << "_incx, tensor_" << fNY << " + out_offset, &" << OpName << "_incy);\n";

      }
      out << SP << "}\n"; // end of batch size loop

      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeW.empty() || fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");

      size_t oDepth  = (fDim > 2) ? fShapeY[2]      : 1;
      size_t oHeight = (fDim > 1) ? fShapeY[fDim]   : 1;
      size_t oWidth  = fShapeY[fDim + 1];
      size_t iDepth  = (fDim > 2) ? fShapeX[2]      : 1;
      size_t iHeight = (fDim > 1) ? fShapeX[fDim]   : 1;
      size_t iWidth  = fShapeX[fDim + 1];
      size_t kHeight = (fDim > 1) ? fShapeW[fDim]   : 1;
      size_t kWidth  = fShapeW[fDim + 1];
      size_t kDepth  = (fDim > 2) ? fShapeW[2]      : 1;

      size_t kernelSize  = fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2];
      size_t colRows     = fShapeW[1] * kernelSize;
      size_t colCols     = oDepth * oHeight * oWidth;
      size_t colElements = colRows * colCols;
      size_t outChannels = fShapeW[0];
      size_t spatialSize = oDepth * oHeight * oWidth;

      // Strides for weight vectorisation
      size_t id = (fDim > 2) ? fDim - 3 : 2;
      size_t ih = (fDim > 1) ? fDim - 2 : 1;
      size_t iw = fDim - 1;
      size_t wstrideDil  = fAttrDilations[iw];
      size_t hstrideDil  = fAttrDilations[ih]  * fAttrKernelShape[iw];
      size_t dstrideDil  = fAttrDilations[id]  * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t icstrideDil = fAttrKernelShape[id] * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t ocstrideDil = fShapeW[1] * icstrideDil;
      size_t hstride     = kWidth;
      size_t dstride     = kHeight * kWidth;
      size_t icstride    = kHeight * kWidth * kDepth;
      size_t ocstride    = fShapeW[1] * icstride;
      size_t wTotalElements = ConvertShapeToLength(fShapeW);

      std::string op;

      // -----------------------------------------------------------------------
      // Kernel 1: Weight vectorisation — reorder W into _f with dilation layout
      // Each thread handles one output element of _f
      // -----------------------------------------------------------------------
      std::string wKname = "WeightVecKernel_" + opName;
      op  = "\n//------ WEIGHT_VEC_KERNEL_ALPAKA (Conv " + opName + ")\n";
      op += SP + "struct " + wKname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ W,\n";
      op += SP + SP + SP + "T* __restrict__ f,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      // Decompose elem_idx into (oc, ic, kd, kh, kw) using compile-time strides
      op += SP + SP + SP + SP + "std::size_t const oc    = elem_idx / " + std::to_string(ocstride) + "u;\n";
      op += SP + SP + SP + SP + "std::size_t const oc_rem = elem_idx % " + std::to_string(ocstride) + "u;\n";
      op += SP + SP + SP + SP + "std::size_t const ic    = oc_rem / " + std::to_string(icstride) + "u;\n";
      op += SP + SP + SP + SP + "std::size_t const ic_rem = oc_rem % " + std::to_string(icstride) + "u;\n";
      if (fDim > 2) {
         op += SP + SP + SP + SP + "std::size_t const kd = ic_rem / " + std::to_string(kHeight * kWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = (ic_rem / " + std::to_string(kWidth) + "u) % " + std::to_string(kHeight) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = ic_rem % " + std::to_string(kWidth) + "u;\n\n";
      } else if (fDim > 1) {
         op += SP + SP + SP + SP + "std::size_t const kd = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = ic_rem / " + std::to_string(kWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = ic_rem % " + std::to_string(kWidth) + "u;\n\n";
      } else {
         op += SP + SP + SP + SP + "std::size_t const kd = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = ic_rem;\n\n";
      }

      // Compute destination index in _f (dilated layout)
      op += SP + SP + SP + SP + "std::size_t const f_idx =\n";
      op += SP + SP + SP + SP + SP + "oc * " + std::to_string(ocstrideDil) + "u +\n";
      op += SP + SP + SP + SP + SP + "ic * " + std::to_string(icstrideDil) + "u";
      if (fDim > 2) op += " +\n" + SP + SP + SP + SP + SP + "kd * " + std::to_string(dstrideDil) + "u";
      if (fDim > 1) op += " +\n" + SP + SP + SP + SP + SP + "kh * " + std::to_string(hstrideDil) + "u";
      op += " +\n" + SP + SP + SP + SP + SP + "kw * " + std::to_string(wstrideDil) + "u;\n\n";

      op += SP + SP + SP + SP + "f[f_idx] = W[elem_idx];\n";
      op += SP + SP + SP + "}\n";   // end grid-stride loop
      op += SP + SP + "}\n";        // end operator()
      op += SP + "};\n\n";          // end struct

      // -----------------------------------------------------------------------
      // Kernel 2: Im2Col
      // -----------------------------------------------------------------------
      std::string im2colKname = "Im2ColKernel_" + opName;
      op += SP + "//------ IM2COL_KERNEL_ALPAKA (Conv " + opName + ")\n";
      op += SP + "struct " + im2colKname + " {\n";
      op += SP + SP + "template<typename TAcc, typename T>\n";
      op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
      op += SP + SP + SP + "TAcc const& acc,\n";
      op += SP + SP + SP + "T const* __restrict__ input,\n";
      op += SP + SP + SP + "T* __restrict__ col,\n";
      op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

      op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
      op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
      op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

      op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n\n";

      op += SP + SP + SP + SP + "std::size_t const col_row = elem_idx / " + std::to_string(colCols) + "u;\n";
      op += SP + SP + SP + SP + "std::size_t const col_col = elem_idx % " + std::to_string(colCols) + "u;\n\n";

      op += SP + SP + SP + SP + "std::size_t const ic    = col_row / " + std::to_string(kernelSize) + "u;\n";
      op += SP + SP + SP + SP + "std::size_t const k_rem = col_row % " + std::to_string(kernelSize) + "u;\n";
      if (fDim > 2) {
         op += SP + SP + SP + SP + "std::size_t const kd = k_rem / " + std::to_string(kHeight * kWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = (k_rem / " + std::to_string(kWidth) + "u) % " + std::to_string(kHeight) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = k_rem % " + std::to_string(kWidth) + "u;\n\n";
      } else if (fDim > 1) {
         op += SP + SP + SP + SP + "std::size_t const kd = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = k_rem / " + std::to_string(kWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = k_rem % " + std::to_string(kWidth) + "u;\n\n";
      } else {
         op += SP + SP + SP + SP + "std::size_t const kd = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kh = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const kw = k_rem;\n\n";
      }

      if (fDim > 2) {
         op += SP + SP + SP + SP + "std::size_t const od = col_col / " + std::to_string(oHeight * oWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const oh = (col_col / " + std::to_string(oWidth) + "u) % " + std::to_string(oHeight) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const ow = col_col % " + std::to_string(oWidth) + "u;\n\n";
      } else if (fDim > 1) {
         op += SP + SP + SP + SP + "std::size_t const od = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const oh = col_col / " + std::to_string(oWidth) + "u;\n";
         op += SP + SP + SP + SP + "std::size_t const ow = col_col % " + std::to_string(oWidth) + "u;\n\n";
      } else {
         op += SP + SP + SP + SP + "std::size_t const od = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const oh = 0u;\n";
         op += SP + SP + SP + SP + "std::size_t const ow = col_col;\n\n";
      }

      op += SP + SP + SP + SP + "int64_t const id_in = static_cast<int64_t>(od * " + std::to_string(fAttrStrides[0])
         + "u + kd * " + std::to_string(fAttrDilations[0]) + "u) - " + std::to_string(fAttrPads[0]) + ";\n";
      op += SP + SP + SP + SP + "int64_t const ih_in = static_cast<int64_t>(oh * " + std::to_string(fAttrStrides[1])
         + "u + kh * " + std::to_string(fAttrDilations[1]) + "u) - " + std::to_string(fAttrPads[1]) + ";\n";
      op += SP + SP + SP + SP + "int64_t const iw_in = static_cast<int64_t>(ow * " + std::to_string(fAttrStrides[2])
         + "u + kw * " + std::to_string(fAttrDilations[2]) + "u) - " + std::to_string(fAttrPads[2]) + ";\n\n";

      op += SP + SP + SP + SP + "bool const in_bounds =\n";
      op += SP + SP + SP + SP + SP + "id_in >= 0 && id_in < " + std::to_string(iDepth)  + " &&\n";
      op += SP + SP + SP + SP + SP + "ih_in >= 0 && ih_in < " + std::to_string(iHeight) + " &&\n";
      op += SP + SP + SP + SP + SP + "iw_in >= 0 && iw_in < " + std::to_string(iWidth)  + ";\n\n";

      op += SP + SP + SP + SP + "if (in_bounds) {\n";
      op += SP + SP + SP + SP + SP + "std::size_t const in_idx =\n";
      op += SP + SP + SP + SP + SP + SP + "ic * " + std::to_string(iDepth * iHeight * iWidth) + "u +\n";
      op += SP + SP + SP + SP + SP + SP + "static_cast<std::size_t>(id_in) * " + std::to_string(iHeight * iWidth) + "u +\n";
      op += SP + SP + SP + SP + SP + SP + "static_cast<std::size_t>(ih_in) * " + std::to_string(iWidth) + "u +\n";
      op += SP + SP + SP + SP + SP + SP + "static_cast<std::size_t>(iw_in);\n";
      op += SP + SP + SP + SP + SP + "col[elem_idx] = input[in_idx];\n";
      op += SP + SP + SP + SP + "} else {\n";
      op += SP + SP + SP + SP + SP + "col[elem_idx] = static_cast<T>(0);\n";
      op += SP + SP + SP + SP + "}\n";
      op += SP + SP + SP + "}\n";
      op += SP + SP + "}\n";
      op += SP + "};\n\n";

      // -----------------------------------------------------------------------
      // Kernel 3: Bias broadcast (only if bias present)
      // -----------------------------------------------------------------------
      if (!fNB2.empty()) {
         std::string biasKname = "BiasBroadcastKernel_" + opName;
         op += SP + "//------ BIAS_BROADCAST_KERNEL_ALPAKA (Conv " + opName + ")\n";
         op += SP + "struct " + biasKname + " {\n";
         op += SP + SP + "template<typename TAcc, typename T>\n";
         op += SP + SP + "ALPAKA_FN_ACC void operator()(\n";
         op += SP + SP + SP + "TAcc const& acc,\n";
         op += SP + SP + SP + "T const* __restrict__ bias,\n";
         op += SP + SP + SP + "T* __restrict__ output,\n";
         op += SP + SP + SP + "std::size_t const totalElements) const {\n\n";

         op += SP + SP + SP + "auto const global_thread_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];\n";
         op += SP + SP + SP + "if (global_thread_idx >= totalElements) return;\n";
         op += SP + SP + SP + "auto const grid_thread_extent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];\n\n";

         op += SP + SP + SP + "for (std::size_t elem_idx = global_thread_idx; elem_idx < totalElements; elem_idx += grid_thread_extent) {\n";
         op += SP + SP + SP + SP + "std::size_t const channel = elem_idx / " + std::to_string(spatialSize) + "u;\n";
         op += SP + SP + SP + SP + "output[elem_idx] = bias[channel];\n";
         op += SP + SP + SP + "}\n";
         op += SP + SP + "}\n";
         op += SP + "};\n\n";
      }

      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      std::string op;
      op  = SP + "WeightVecKernel_"  + opName + " weightVecKernel_"  + opName + ";\n";
      op += SP + "Im2ColKernel_"     + opName + " im2colKernel_"     + opName + ";\n";
      if (!fNB2.empty())
         op += SP + "BiasBroadcastKernel_" + opName + " biasBroadcastKernel_" + opName + ";\n";
      return op;
   }

   std::string Generate_GPU_ALPAKA(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeW.empty() || fShapeY.empty())
         throw std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");

      size_t bsize       = fShapeX[0];
      size_t oDepth      = (fDim > 2) ? fShapeY[2]    : 1;
      size_t oHeight     = (fDim > 1) ? fShapeY[fDim] : 1;
      size_t oWidth      = fShapeY[fDim + 1];
      size_t iDepth      = (fDim > 2) ? fShapeX[2]    : 1;
      size_t iHeight     = (fDim > 1) ? fShapeX[fDim] : 1;
      size_t iWidth      = fShapeX[fDim + 1];
      size_t outChannels = fShapeW[0];
      size_t kernelSize  = fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2];
      size_t colElements = gemm_k * gemm_m;   // colRows * colCols
      size_t wTotal      = ConvertShapeToLength(fShapeW);

      // For group conv: per-group output channels and _f offset
      // gemm_n stays as total output channels — we divide per group at launch
      size_t groupFOffset     = gemm_n * gemm_k;  // elements of _f per group

      std::stringstream out;
      out << "\n//------ CONV_GPU_ALPAKA\n";

      // -----------------------------------------------------------------------
      // Step 1: Weight vectorisation kernel — runs once, fully on GPU
      // -----------------------------------------------------------------------
      out << SP << "// Step 1: vectorise W -> _f on GPU (once per infer call)\n";
      out << SP << "{\n";
      out << SP << SP << "auto const elementsPerThread_wv = Vec::all(static_cast<Idx>(1));\n";
      out << SP << SP << "auto const elementsPerGrid_wv   = Vec::all(Idx{" << wTotal << "});\n";
      out << SP << SP << "alpaka::KernelCfg<Acc> const cfg_wv = {elementsPerGrid_wv, elementsPerThread_wv};\n";
      out << SP << SP << "auto const workDiv_wv = alpaka::getValidWorkDiv(cfg_wv, devAcc, weightVecKernel_" << opName
         << ", alpaka::getPtrNative(deviceBuf_" << fNW << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << convK << ")"
         << ", static_cast<Idx>(" << wTotal << "));\n";
      out << SP << SP << "alpaka::exec<Acc>(queue, workDiv_wv, weightVecKernel_" << opName
         << ", alpaka::getPtrNative(deviceBuf_" << fNW << ")"
         << ", alpaka::getPtrNative(deviceBuf_" << convK << ")"
         << ", static_cast<Idx>(" << wTotal << "));\n";
      out << SP << SP << "alpaka::wait(queue);\n";
      out << SP << "}\n\n";

      // -----------------------------------------------------------------------
      // Step 2: Batch loop
      // -----------------------------------------------------------------------
      out << SP << "for (std::size_t n = 0; n < " << bsize << "; n++) {\n\n";
      out << SP << SP << "std::size_t const x_offset   = n * "
         << fShapeX[1] * iDepth * iHeight * iWidth << "u;\n";
      out << SP << SP << "std::size_t const out_offset = n * "
         << fShapeY[1] * gemm_m << "u;\n\n";

      // -----------------------------------------------------------------------
      // Step 3: Im2Col
      // -----------------------------------------------------------------------
      out << SP << SP << "// Step 3: im2col\n";
      out << SP << SP << "{\n";
      out << SP << SP << SP << "auto const elementsPerThread_im2col = Vec::all(static_cast<Idx>(1));\n";
      out << SP << SP << SP << "auto const elementsPerGrid_im2col   = Vec::all(Idx{" << colElements << "});\n";
      out << SP << SP << SP << "alpaka::KernelCfg<Acc> const cfg_im2col = {elementsPerGrid_im2col, elementsPerThread_im2col};\n";
      out << SP << SP << SP << "auto const workDiv_im2col = alpaka::getValidWorkDiv(cfg_im2col, devAcc, im2colKernel_" << opName
         << ", alpaka::getPtrNative(deviceBuf_" << fNX << ") + x_offset"
         << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
         << ", static_cast<Idx>(" << colElements << "));\n";
      out << SP << SP << SP << "alpaka::exec<Acc>(queue, workDiv_im2col, im2colKernel_" << opName
         << ", alpaka::getPtrNative(deviceBuf_" << fNX << ") + x_offset"
         << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
         << ", static_cast<Idx>(" << colElements << "));\n";
      out << SP << SP << SP << "alpaka::wait(queue);\n";
      out << SP << SP << "}\n\n";

      // -----------------------------------------------------------------------
      // Step 4: GEMM (+ optional bias) — group or non-group
      // gemm_n/gemm_m/gemm_k are member variables set in Initialize()
      // For grouped conv we use gemm_n/fAttrGroup per group, keeping gemm_n as total
      // -----------------------------------------------------------------------
      if (fAttrGroup == 1) {
         if (!fNB2.empty()) {
               size_t biasElements = gemm_n * gemm_m;
               out << SP << SP << "// Step 4a: broadcast bias into output slice\n";
               out << SP << SP << "{\n";
               out << SP << SP << SP << "auto const elementsPerThread_bias = Vec::all(static_cast<Idx>(1));\n";
               out << SP << SP << SP << "auto const elementsPerGrid_bias   = Vec::all(Idx{" << biasElements << "});\n";
               out << SP << SP << SP << "alpaka::KernelCfg<Acc> const cfg_bias = {elementsPerGrid_bias, elementsPerThread_bias};\n";
               out << SP << SP << SP << "auto const workDiv_bias = alpaka::getValidWorkDiv(cfg_bias, devAcc, biasBroadcastKernel_" << opName
                  << ", alpaka::getPtrNative(deviceBuf_" << fNB2 << ")"
                  << ", alpaka::getPtrNative(deviceBuf_" << fNY << ") + out_offset"
                  << ", static_cast<Idx>(" << biasElements << "));\n";
               out << SP << SP << SP << "alpaka::exec<Acc>(queue, workDiv_bias, biasBroadcastKernel_" << opName
                  << ", alpaka::getPtrNative(deviceBuf_" << fNB2 << ")"
                  << ", alpaka::getPtrNative(deviceBuf_" << fNY << ") + out_offset"
                  << ", static_cast<Idx>(" << biasElements << "));\n";
               out << SP << SP << SP << "alpaka::wait(queue);\n";
               out << SP << SP << "}\n\n";
               out << SP << SP << "// Step 4b: GEMM beta=1 fuses bias\n";
               out << SP << SP << "blas.matmul('n', 'n', "
                  << gemm_n << ", " << gemm_m << ", " << gemm_k
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << convK << ")"
                  << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << fNY << ") + out_offset);\n\n";
         } else {
               out << SP << SP << "// Step 4: GEMM beta=0 (no bias)\n";
               out << SP << SP << "blas.matmul('n', 'n', "
                  << gemm_n << ", " << gemm_m << ", " << gemm_k
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << convK << ")"
                  << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
                  << ", 0.0f, alpaka::getPtrNative(deviceBuf_" << fNY << ") + out_offset);\n\n";
         }
      } else {
         // Group convolution — gemm_n stays as total, divide per group at launch
         out << SP << SP << "for (std::size_t g = 0; g < " << fAttrGroup << "; g++) {\n\n";
         out << SP << SP << SP << "std::size_t const g_out_offset = out_offset + g * "
               << gemm_n * gemm_m << "u;\n";
         out << SP << SP << SP << "std::size_t const f_offset = g * " << groupFOffset << "u;\n\n";

         if (!fNB2.empty()) {
               size_t groupBiasElements = gemm_n * gemm_m;
               out << SP << SP << SP << "// Broadcast group bias\n";
               out << SP << SP << SP << "{\n";
               out << SP << SP << SP << SP << "auto const elementsPerThread_bias = Vec::all(static_cast<Idx>(1));\n";
               out << SP << SP << SP << SP << "auto const elementsPerGrid_bias   = Vec::all(Idx{" << groupBiasElements << "});\n";
               out << SP << SP << SP << SP << "alpaka::KernelCfg<Acc> const cfg_bias = {elementsPerGrid_bias, elementsPerThread_bias};\n";
               out << SP << SP << SP << SP << "auto const workDiv_bias = alpaka::getValidWorkDiv(cfg_bias, devAcc, biasBroadcastKernel_" << opName
                  << ", alpaka::getPtrNative(deviceBuf_" << fNB2 << ") + g * " << gemm_n
                  << ", alpaka::getPtrNative(deviceBuf_" << fNY << ") + g_out_offset"
                  << ", static_cast<Idx>(" << groupBiasElements << "));\n";
               out << SP << SP << SP << SP << "alpaka::exec<Acc>(queue, workDiv_bias, biasBroadcastKernel_" << opName
                  << ", alpaka::getPtrNative(deviceBuf_" << fNB2 << ") + g * " << gemm_n
                  << ", alpaka::getPtrNative(deviceBuf_" << fNY << ") + g_out_offset"
                  << ", static_cast<Idx>(" << groupBiasElements << "));\n";
               out << SP << SP << SP << SP << "alpaka::wait(queue);\n";
               out << SP << SP << SP << "}\n\n";
               out << SP << SP << SP << "blas.matmul('n', 'n', "
                  << gemm_n << ", " << gemm_m << ", " << gemm_k
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << convK << ") + f_offset"
                  << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << fNY << ") + g_out_offset);\n\n";
         } else {
               out << SP << SP << SP << "blas.matmul('n', 'n', "
                  << gemm_n << ", " << gemm_m << ", " << gemm_k
                  << ", 1.0f, alpaka::getPtrNative(deviceBuf_" << convK << ") + f_offset"
                  << ", alpaka::getPtrNative(deviceBuf_" << imcol << ")"
                  << ", 0.0f, alpaka::getPtrNative(deviceBuf_" << fNY << ") + g_out_offset);\n\n";
         }
         out << SP << SP << "}\n"; // end group loop
      }
      
      // print the contents of deviceBuf_W, deviceBuf_x_f and deviceBuf_x_xcol for debugging
      out << SP << SP << "}\n";
      out << SP << SP << "// Debug: print _W, _f and _xcol\n";
      out << SP << SP << "std::cout << \"_W: \";\n";
      out << SP << SP << "for (std::size_t i = 0; i < " << wTotal << "; i++) {\n";
      out << SP << SP << SP << "printf(\"%f \", alpaka::getPtrNative(deviceBuf_" << fNW << ")  + i);\n";
      out << SP << SP << "std::cout << std::endl;}\n";
      out << SP << SP << "for (std::size_t i = 0; i < " << wTotal << "; i++) {\n";
      out << SP << SP << SP << "printf(\"%f \", alpaka::getPtrNative(deviceBuf_" << convK << ")  + i);\n";
      out << SP << SP << "}\n";
      out << SP << SP << "std::cout << std::endl;\n";
      out << SP << SP << "std::cout << \"_xcol: \";\n";
      out << SP << SP << "for (std::size_t i = 0; i < " << colElements << "; i++) {\n";
      out << SP << SP << SP << "printf(\"%f \", alpaka::getPtrNative(deviceBuf_" << imcol << ")  + i);\n";
      out << SP << SP << "std::cout << std::endl;\n"; 


      out << SP << "}\n"; // end batch loop
      return out.str();
   }

   /*! \brief Returns the blas routines needed to compile the generated code
    */
   std::vector<std::string> GetBlasRoutines() override { return { std::string("Gemm"), std::string("Axpy") }; }


   std::string GetBlasConfig(){
      auto lda = std::to_string(gemm_k);
      auto ldb = std::to_string(gemm_n);
      auto ldc = std::to_string(gemm_n);
      return std::to_string(gemm_n) + ", " + std::to_string(gemm_m) + ", " + std::to_string(gemm_k) + ", " + ldb + ", " + lda + ", " + ldc;
   }

};

} // namespace SOFIE

#endif
