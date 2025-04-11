//Code generated automatically by TMVA for GPU Inference using ALPAKA of Model file [Linear_16.onnx] at [Fri Apr 11 14:16:45 2025] 

#ifndef SOFIE_LINEAR_16
#define SOFIE_LINEAR_16

#include <algorithm>
#include <vector>
#include <alpaka/alpaka.hpp>
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>
#include "SOFIE/SOFIE_common.hxx"
#include <fstream>

using Dim1D = alpaka::DimInt<1>;
using Acc = alpaka::TagToAcc<alpaka::TagGpuCudaRt, Dim1D, Idx>;
using Queue = alpaka::Queue<Acc, alpaka::Blocking>;

namespace SOFIE_Linear_16{
struct Session {

// initialized tensors
auto deviceBuf_8weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_8bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_4bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_2weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_0bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_12bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_18bias = alpaka::allocBuf<float, size_t>(devAcc, 10);
auto deviceBuf_14bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_4weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_10weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_6bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_18weight = alpaka::allocBuf<float, size_t>(devAcc, 500);
auto deviceBuf_0weight = alpaka::allocBuf<float, size_t>(devAcc, 5000);
auto deviceBuf_10bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_2bias = alpaka::allocBuf<float, size_t>(devAcc, 50);
auto deviceBuf_6weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_14weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_16weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_12weight = alpaka::allocBuf<float, size_t>(devAcc, 2500);
auto deviceBuf_16bias = alpaka::allocBuf<float, size_t>(devAcc, 50);

//--- declare and allocate the intermediate tensors
auto bufDev_18biasbcast = alpaka::allocBuf<float, size_t>(devAcc,160);
auto bufDev_38 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_14biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_34 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_22 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_2biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_24 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_0biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_6biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_4biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_16biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_8biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_26 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_28 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_10biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_30 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_32 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_36 = alpaka::allocBuf<float, size_t>(devAcc,800);
auto bufDev_12biasbcast = alpaka::allocBuf<float, size_t>(devAcc,800);

Session(std::string filename ="Linear_16.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   std::string tensor_name;
   size_t length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_8weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_8weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_8weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_8weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_8bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_8bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_8bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_8bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_4bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_4bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_4bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_4bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_2weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_2weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_2weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_2weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_0bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_0bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_0bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_0bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_12bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_12bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_12bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_12bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_18bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_18bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 10) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 10 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_18bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_18bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_14bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_14bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_14bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_14bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_4weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_4weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_4weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_4weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_10weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_10weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_10weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_10weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_6bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_6bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_6bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_6bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_18weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_18weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_18weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_18weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_0weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_0weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 5000) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 5000 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_0weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_0weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_10bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_10bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_10bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_10bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_2bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_2bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_2bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_2bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_6weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_6weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_6weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_6weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_14weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_14weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_14weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_14weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_16weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_16weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_16weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_16weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_12weight" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_12weight , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 2500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 2500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_12weight[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_12weight");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_16bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_16bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_16bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_16bias");
   }
   f.close();

   auto hostBuf_8weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_8weight), tensor_8weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_8weight, hostBuf8weight, 2500);
   auto hostBuf_8bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_8bias), tensor_8bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_8bias, hostBuf8bias, 50);
   auto hostBuf_4bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_4bias), tensor_4bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_4bias, hostBuf4bias, 50);
   auto hostBuf_2weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_2weight), tensor_2weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_2weight, hostBuf2weight, 2500);
   auto hostBuf_0bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_0bias), tensor_0bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_0bias, hostBuf0bias, 50);
   auto hostBuf_12bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_12bias), tensor_12bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_12bias, hostBuf12bias, 50);
   auto hostBuf_18bias = alpaka::allocBuf<DataType, Idx>(hostAcc,10);
   std::memcpy(alpaka::getPtrNative(hostBuf_18bias), tensor_18bias, 10* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_18bias, hostBuf18bias, 10);
   auto hostBuf_14bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_14bias), tensor_14bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_14bias, hostBuf14bias, 50);
   auto hostBuf_4weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_4weight), tensor_4weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_4weight, hostBuf4weight, 2500);
   auto hostBuf_10weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_10weight), tensor_10weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_10weight, hostBuf10weight, 2500);
   auto hostBuf_6bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_6bias), tensor_6bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_6bias, hostBuf6bias, 50);
   auto hostBuf_18weight = alpaka::allocBuf<DataType, Idx>(hostAcc,500);
   std::memcpy(alpaka::getPtrNative(hostBuf_18weight), tensor_18weight, 500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_18weight, hostBuf18weight, 500);
   auto hostBuf_0weight = alpaka::allocBuf<DataType, Idx>(hostAcc,5000);
   std::memcpy(alpaka::getPtrNative(hostBuf_0weight), tensor_0weight, 5000* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_0weight, hostBuf0weight, 5000);
   auto hostBuf_10bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_10bias), tensor_10bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_10bias, hostBuf10bias, 50);
   auto hostBuf_2bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_2bias), tensor_2bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_2bias, hostBuf2bias, 50);
   auto hostBuf_6weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_6weight), tensor_6weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_6weight, hostBuf6weight, 2500);
   auto hostBuf_14weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_14weight), tensor_14weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_14weight, hostBuf14weight, 2500);
   auto hostBuf_16weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_16weight), tensor_16weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_16weight, hostBuf16weight, 2500);
   auto hostBuf_12weight = alpaka::allocBuf<DataType, Idx>(hostAcc,2500);
   std::memcpy(alpaka::getPtrNative(hostBuf_12weight), tensor_12weight, 2500* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_12weight, hostBuf12weight, 2500);
   auto hostBuf_16bias = alpaka::allocBuf<DataType, Idx>(hostAcc,50);
   std::memcpy(alpaka::getPtrNative(hostBuf_16bias), tensor_16bias, 50* sizeof(float));
   alpaka::memcpy(queue, deviceBuf_16bias, hostBuf16bias, 50);

//---- allocate the intermediate dynamic tensors
//--- broadcast bias tensor 0biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_0bias,{ 50 }, { 16 , 50 });
      auto hostBuf_0biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_0biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_0biasbcast, hostBuf_0biasbcast , 800);
   }
//--- broadcast bias tensor 2biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_2bias,{ 50 }, { 16 , 50 });
      auto hostBuf_2biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_2biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_2biasbcast, hostBuf_2biasbcast , 800);
   }
//--- broadcast bias tensor 4biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_4bias,{ 50 }, { 16 , 50 });
      auto hostBuf_4biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_4biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_4biasbcast, hostBuf_4biasbcast , 800);
   }
//--- broadcast bias tensor 6biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_6bias,{ 50 }, { 16 , 50 });
      auto hostBuf_6biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_6biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_6biasbcast, hostBuf_6biasbcast , 800);
   }
//--- broadcast bias tensor 8biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_8bias,{ 50 }, { 16 , 50 });
      auto hostBuf_8biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_8biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_8biasbcast, hostBuf_8biasbcast , 800);
   }
//--- broadcast bias tensor 10biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_10bias,{ 50 }, { 16 , 50 });
      auto hostBuf_10biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_10biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_10biasbcast, hostBuf_10biasbcast , 800);
   }
//--- broadcast bias tensor 12biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_12bias,{ 50 }, { 16 , 50 });
      auto hostBuf_12biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_12biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_12biasbcast, hostBuf_12biasbcast , 800);
   }
//--- broadcast bias tensor 14biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_14bias,{ 50 }, { 16 , 50 });
      auto hostBuf_14biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_14biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_14biasbcast, hostBuf_14biasbcast , 800);
   }
//--- broadcast bias tensor 16biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_16bias,{ 50 }, { 16 , 50 });
      auto hostBuf_16biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,800);
      std::memcpy(alpaka::getPtrNative(hostBuf_16biasbcast), data, 800 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_16biasbcast, hostBuf_16biasbcast , 800);
   }
//--- broadcast bias tensor 18biasfor Gemm op
   {
      float * data = SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_18bias,{ 10 }, { 16 , 10 });
      auto hostBuf_18biasbcast = alpaka::allocBuf<float, size_t>(hostAcc,160);
      std::memcpy(alpaka::getPtrNative(hostBuf_18biasbcast), data, 160 * sizeof(float));
      alpaka::memcpy(queue, deviceBuf_18biasbcast, hostBuf_18biasbcast , 160);
   }
}



std::vector<float> infer(float* tensor_input1){

//--------- Gemm_GPU_ALPAKA
   char op_0_transA = 'n';
   char op_0_transB = 't';
   int op_0_m = 16;
   int op_0_n = 50;
   int op_0_k = 100;
   float op_0_alpha = 1;
   float op_0_beta = 1;
   int op_0_lda = 100;
   int op_0_ldb = 100;
   std::copy(tensor_0biasbcast, tensor_0biasbcast + 800, tensor_22);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_input1((float*)std::data(bufDev_input1), op_0_m, op_0_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_0weight((float*)std::data(bufDev_0weight), op_0_k, op_0_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_22((float*)std::data(bufDev_22), op_0_m, op_0_n);
   KokkosBlas::gemm(&op_0_transB, &op_0_transA, op_0_alpha, kokkos_dev_input1, kokkos_dev_0weight, op_0_beta, kokkos_dev_22);

//--------- Gemm_GPU_ALPAKA
   char op_1_transA = 'n';
   char op_1_transB = 't';
   int op_1_m = 16;
   int op_1_n = 50;
   int op_1_k = 50;
   float op_1_alpha = 1;
   float op_1_beta = 1;
   int op_1_lda = 50;
   int op_1_ldb = 50;
   std::copy(tensor_2biasbcast, tensor_2biasbcast + 800, tensor_24);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_22((float*)std::data(bufDev_22), op_1_m, op_1_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_2weight((float*)std::data(bufDev_2weight), op_1_k, op_1_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_24((float*)std::data(bufDev_24), op_1_m, op_1_n);
   KokkosBlas::gemm(&op_1_transB, &op_1_transA, op_1_alpha, kokkos_dev_22, kokkos_dev_2weight, op_1_beta, kokkos_dev_24);

//--------- Gemm_GPU_ALPAKA
   char op_2_transA = 'n';
   char op_2_transB = 't';
   int op_2_m = 16;
   int op_2_n = 50;
   int op_2_k = 50;
   float op_2_alpha = 1;
   float op_2_beta = 1;
   int op_2_lda = 50;
   int op_2_ldb = 50;
   std::copy(tensor_4biasbcast, tensor_4biasbcast + 800, tensor_26);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_24((float*)std::data(bufDev_24), op_2_m, op_2_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_4weight((float*)std::data(bufDev_4weight), op_2_k, op_2_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_26((float*)std::data(bufDev_26), op_2_m, op_2_n);
   KokkosBlas::gemm(&op_2_transB, &op_2_transA, op_2_alpha, kokkos_dev_24, kokkos_dev_4weight, op_2_beta, kokkos_dev_26);

//--------- Gemm_GPU_ALPAKA
   char op_3_transA = 'n';
   char op_3_transB = 't';
   int op_3_m = 16;
   int op_3_n = 50;
   int op_3_k = 50;
   float op_3_alpha = 1;
   float op_3_beta = 1;
   int op_3_lda = 50;
   int op_3_ldb = 50;
   std::copy(tensor_6biasbcast, tensor_6biasbcast + 800, tensor_28);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_26((float*)std::data(bufDev_26), op_3_m, op_3_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_6weight((float*)std::data(bufDev_6weight), op_3_k, op_3_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_28((float*)std::data(bufDev_28), op_3_m, op_3_n);
   KokkosBlas::gemm(&op_3_transB, &op_3_transA, op_3_alpha, kokkos_dev_26, kokkos_dev_6weight, op_3_beta, kokkos_dev_28);

//--------- Gemm_GPU_ALPAKA
   char op_4_transA = 'n';
   char op_4_transB = 't';
   int op_4_m = 16;
   int op_4_n = 50;
   int op_4_k = 50;
   float op_4_alpha = 1;
   float op_4_beta = 1;
   int op_4_lda = 50;
   int op_4_ldb = 50;
   std::copy(tensor_8biasbcast, tensor_8biasbcast + 800, tensor_30);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_28((float*)std::data(bufDev_28), op_4_m, op_4_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_8weight((float*)std::data(bufDev_8weight), op_4_k, op_4_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_30((float*)std::data(bufDev_30), op_4_m, op_4_n);
   KokkosBlas::gemm(&op_4_transB, &op_4_transA, op_4_alpha, kokkos_dev_28, kokkos_dev_8weight, op_4_beta, kokkos_dev_30);

//--------- Gemm_GPU_ALPAKA
   char op_5_transA = 'n';
   char op_5_transB = 't';
   int op_5_m = 16;
   int op_5_n = 50;
   int op_5_k = 50;
   float op_5_alpha = 1;
   float op_5_beta = 1;
   int op_5_lda = 50;
   int op_5_ldb = 50;
   std::copy(tensor_10biasbcast, tensor_10biasbcast + 800, tensor_32);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_30((float*)std::data(bufDev_30), op_5_m, op_5_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_10weight((float*)std::data(bufDev_10weight), op_5_k, op_5_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_32((float*)std::data(bufDev_32), op_5_m, op_5_n);
   KokkosBlas::gemm(&op_5_transB, &op_5_transA, op_5_alpha, kokkos_dev_30, kokkos_dev_10weight, op_5_beta, kokkos_dev_32);

//--------- Gemm_GPU_ALPAKA
   char op_6_transA = 'n';
   char op_6_transB = 't';
   int op_6_m = 16;
   int op_6_n = 50;
   int op_6_k = 50;
   float op_6_alpha = 1;
   float op_6_beta = 1;
   int op_6_lda = 50;
   int op_6_ldb = 50;
   std::copy(tensor_12biasbcast, tensor_12biasbcast + 800, tensor_34);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_32((float*)std::data(bufDev_32), op_6_m, op_6_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_12weight((float*)std::data(bufDev_12weight), op_6_k, op_6_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_34((float*)std::data(bufDev_34), op_6_m, op_6_n);
   KokkosBlas::gemm(&op_6_transB, &op_6_transA, op_6_alpha, kokkos_dev_32, kokkos_dev_12weight, op_6_beta, kokkos_dev_34);

//--------- Gemm_GPU_ALPAKA
   char op_7_transA = 'n';
   char op_7_transB = 't';
   int op_7_m = 16;
   int op_7_n = 50;
   int op_7_k = 50;
   float op_7_alpha = 1;
   float op_7_beta = 1;
   int op_7_lda = 50;
   int op_7_ldb = 50;
   std::copy(tensor_14biasbcast, tensor_14biasbcast + 800, tensor_36);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_34((float*)std::data(bufDev_34), op_7_m, op_7_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_14weight((float*)std::data(bufDev_14weight), op_7_k, op_7_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_36((float*)std::data(bufDev_36), op_7_m, op_7_n);
   KokkosBlas::gemm(&op_7_transB, &op_7_transA, op_7_alpha, kokkos_dev_34, kokkos_dev_14weight, op_7_beta, kokkos_dev_36);

//--------- Gemm_GPU_ALPAKA
   char op_8_transA = 'n';
   char op_8_transB = 't';
   int op_8_m = 16;
   int op_8_n = 50;
   int op_8_k = 50;
   float op_8_alpha = 1;
   float op_8_beta = 1;
   int op_8_lda = 50;
   int op_8_ldb = 50;
   std::copy(tensor_16biasbcast, tensor_16biasbcast + 800, tensor_38);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_36((float*)std::data(bufDev_36), op_8_m, op_8_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_16weight((float*)std::data(bufDev_16weight), op_8_k, op_8_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_38((float*)std::data(bufDev_38), op_8_m, op_8_n);
   KokkosBlas::gemm(&op_8_transB, &op_8_transA, op_8_alpha, kokkos_dev_36, kokkos_dev_16weight, op_8_beta, kokkos_dev_38);

//--------- Gemm_GPU_ALPAKA
   char op_9_transA = 'n';
   char op_9_transB = 't';
   int op_9_m = 16;
   int op_9_n = 10;
   int op_9_k = 50;
   float op_9_alpha = 1;
   float op_9_beta = 1;
   int op_9_lda = 50;
   int op_9_ldb = 50;
   std::copy(tensor_18biasbcast, tensor_18biasbcast + 160, tensor_39);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_38((float*)std::data(bufDev_38), op_9_m, op_9_k);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_18weight((float*)std::data(bufDev_18weight), op_9_k, op_9_n);
   Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::CudaSpace> kokkos_dev_39((float*)std::data(bufDev_39), op_9_m, op_9_n);
   KokkosBlas::gemm(&op_9_transB, &op_9_transA, op_9_alpha, kokkos_dev_38, kokkos_dev_18weight, op_9_beta, kokkos_dev_39);
   return {std::vector<float>(tensor_39, tensor_39 + 160)};
}
};   // end of Session
} //SOFIE_Linear_16

#endif  // SOFIE_LINEAR_16
