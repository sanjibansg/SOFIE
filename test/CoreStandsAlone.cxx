// Base SOFIE without the quantization package: this translation unit links against
// SOFIE_core alone, so it fails the moment base references a symbol the package defines.

#include "SOFIE/RModel.hxx"
#include "SOFIE/RModelCodegenPass.hxx"
#include "SOFIE/ROperator_Relu.hxx"

#include <cstdio>
#include <memory>
#include <vector>

int main()
{
   if (SOFIE::InstalledCodegenPass() != nullptr) {
      std::printf("a codegen pass is installed; base was expected to link without one\n");
      return 1;
   }

   SOFIE::RModel model("CoreStandsAlone", "");
   model.AddInputTensorInfo("x", SOFIE::ETensorType::FLOAT, std::vector<std::size_t>{4});
   model.AddInputTensorName("x");
   model.AddOperator(std::make_unique<SOFIE::ROperator_Relu<float>>("x", "y"));
   model.AddOutputTensorNameList({"y"});
   model.Generate();

   if (model.ReturnGenerated().empty()) {
      std::printf("generation produced nothing with no pass installed\n");
      return 1;
   }
   return 0;
}
