#include "SOFIE/RModelCodegenPass.hxx"

namespace SOFIE {

namespace {
std::unique_ptr<RModelCodegenPass> &CodegenPassSlot()
{
   static std::unique_ptr<RModelCodegenPass> slot;
   return slot;
}
} // namespace

void InstallCodegenPass(std::unique_ptr<RModelCodegenPass> pass)
{
   CodegenPassSlot() = std::move(pass);
}

RModelCodegenPass *InstalledCodegenPass()
{
   return CodegenPassSlot().get();
}

} // namespace SOFIE
