// Reference output for Pad.onnx ([1,2,2] -> [2,3,5], constant mode, pads before=[1,0,1] after=[0,1,2])
// Expected computed with numpy.pad
#pragma once
namespace Pad_ExpectedOutput {
   static float outputs[30] = {
      0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f,
      0.f, 1.f, 2.f, 0.f, 0.f,
      0.f, 3.f, 4.f, 0.f, 0.f,
      0.f, 0.f, 0.f, 0.f, 0.f
   };
} // namespace Pad_ExpectedOutput
