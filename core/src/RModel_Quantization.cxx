#include "SOFIE/RModel.hxx"
#include "SOFIE/RQuantization.hxx"
#include "SOFIE/RQuantization_Pipeline.hxx"
#include "SOFIE/RQuantization_Analysis.hxx"
#include "SOFIE/RQuantization_Convolution.hxx"
#include "SOFIE/RQuantization_DenseLinear.hxx"
#include "SOFIE/RQuantization_Elementwise.hxx"
#include "SOFIE/RQuantization_Gather.hxx"
#include "SOFIE/RQuantization_Storage.hxx"
#include "SOFIE/ROperator_Clip.hxx"
#include "SOFIE/ROperator_ONNXQuantizeLinear.hxx"
#include "SOFIE/ROperator_Reshape.hxx"
#include "SOFIE/ROperator_Softmax.hxx"
#include "SOFIE/ROperator_Transpose.hxx"

#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SOFIE {


QuantizationExtension &QuantizationExtension::Of(RModel &model)
{
   if (auto *attached = dynamic_cast<QuantizationExtension *>(model.GetExtension()))
      return *attached;
   auto owned = std::make_unique<QuantizationExtension>();
   auto &extension = *owned;
   model.SetExtension(std::move(owned));
   return extension;
}

const QuantizationExtension &QuantizationExtension::Of(const RModel &model)
{
   static const QuantizationExtension empty;
   if (const auto *attached = dynamic_cast<const QuantizationExtension *>(model.GetExtension()))
      return *attached;
   return empty;
}

const QuantizationModelState &RModel::GetQuantizationState() const
{
   return QuantizationExtension::Of(*this).state;
}

const QuantizationPipelineReport &RModel::GetQuantizationPipelineReport() const
{
   return QuantizationExtension::Of(*this).report;
}

const std::vector<CarrierFrontierViolation> &RModel::GetCarrierFrontierViolations() const
{
   return QuantizationExtension::Of(*this).carrierFrontierViolations;
}

void RModel::AddQuantizationInfo(const std::string & tensor_name, QuantizationInfo info)
{
   QuantizationExtension::Of(*this).state.tensorInfos[UTILITY::Clean_name(tensor_name)] = std::move(info);
}

bool RModel::HasQuantizationInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   if (QuantizationExtension::Of(*this).state.tensorInfos.find(clean_name) != QuantizationExtension::Of(*this).state.tensorInfos.end())
      return true;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->HasQuantizationInfo(clean_name);
   return false;
}

const QuantizationInfo & RModel::GetQuantizationInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   auto f = QuantizationExtension::Of(*this).state.tensorInfos.find(clean_name);
   if (f != QuantizationExtension::Of(*this).state.tensorInfos.end())
      return f->second;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetQuantizationInfo(clean_name);
   throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no quantization information");
}

void RModel::AddLowPrecisionTensorInfo(const std::string & tensor_name, LowPrecisionTensorInfo info)
{
   QuantizationExtension::Of(*this).state.lowPrecisionTensorInfos[UTILITY::Clean_name(tensor_name)] = std::move(info);
}

bool RModel::HasLowPrecisionTensorInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   if (QuantizationExtension::Of(*this).state.lowPrecisionTensorInfos.find(clean_name) != QuantizationExtension::Of(*this).state.lowPrecisionTensorInfos.end())
      return true;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->HasLowPrecisionTensorInfo(clean_name);
   return false;
}

const LowPrecisionTensorInfo & RModel::GetLowPrecisionTensorInfo(const std::string & tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(tensor_name);
   auto f = QuantizationExtension::Of(*this).state.lowPrecisionTensorInfos.find(clean_name);
   if (f != QuantizationExtension::Of(*this).state.lowPrecisionTensorInfos.end())
      return f->second;
   if (fIsSubGraph && fParentGraph)
      return fParentGraph->GetLowPrecisionTensorInfo(clean_name);
   throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no low-precision carrier information");
}

void RModel::RegisterQuantizedTensorStorage(QuantizedTensorStorage storage)
{
   storage.storageTensor = UTILITY::Clean_name(storage.storageTensor);
   storage.logicalTensor = UTILITY::Clean_name(storage.logicalTensor);
   storage.sourceTensor = UTILITY::Clean_name(storage.sourceTensor);
   if (storage.storageTensor.empty()) {
      throw std::runtime_error("SOFIE quantized tensor storage registration requires a storage tensor name");
   }
   QuantizationExtension::Of(*this).state.tensorStorages[storage.storageTensor] = std::move(storage);
}

bool RModel::HasQuantizedTensorStorage(const std::string & storage_tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(storage_tensor_name);
   return QuantizationExtension::Of(*this).state.tensorStorages.find(clean_name) != QuantizationExtension::Of(*this).state.tensorStorages.end();
}

const QuantizedTensorStorage & RModel::GetQuantizedTensorStorage(const std::string & storage_tensor_name) const
{
   auto clean_name = UTILITY::Clean_name(storage_tensor_name);
   auto it = QuantizationExtension::Of(*this).state.tensorStorages.find(clean_name);
   if (it == QuantizationExtension::Of(*this).state.tensorStorages.end()) {
      throw std::runtime_error("SOFIE tensor [" + clean_name + "] has no quantized storage information");
   }
   return it->second;
}

void RModel::AnalyzeQuantizedRegions()
{
   for (const auto &[name, storage] : QuantizationExtension::Of(*this).state.tensorStorages)
      fInitializedTensors.erase(name);
   QuantizationExtension::Of(*this).state.ClearDerivedAnalysis();

   // Value-preserving operators carry quantization metadata forward: aliases copy it, layout
   // permutations remap per-axis contracts, multi-input aliases require all inputs to agree.
   auto sameQuantizationInfo = [](const QuantizationInfo &lhs, const QuantizationInfo &rhs) {
      return lhs.bitWidth == rhs.bitWidth && lhs.isSigned == rhs.isSigned && lhs.narrow == rhs.narrow &&
             lhs.scale == rhs.scale && lhs.zeroPoint == rhs.zeroPoint && lhs.scaleTensor == rhs.scaleTensor &&
             lhs.zeroPointTensor == rhs.zeroPointTensor && lhs.rounding == rhs.rounding &&
             lhs.overflow == rhs.overflow && lhs.granularity == rhs.granularity && lhs.axis == rhs.axis;
   };

   auto sameLowPrecisionTensorInfo = [&sameQuantizationInfo](const LowPrecisionTensorInfo &lhs,
                                                             const LowPrecisionTensorInfo &rhs) {
      if (lhs.carrier != rhs.carrier ||
          lhs.affineQuantization.has_value() != rhs.affineQuantization.has_value())
         return false;
      if (lhs.affineQuantization)
         return sameQuantizationInfo(*lhs.affineQuantization, *rhs.affineQuantization);
      return true;
   };

   auto remapQuantization = [](QuantizationInfo info, const std::vector<int_t> &axisMap)
      -> std::optional<QuantizationInfo> {
      if (axisMap.empty() || info.granularity == EQuantizationGranularity::PerTensor || info.axis < 0)
         return info;

      auto axis = static_cast<std::size_t>(info.axis);
      for (std::size_t outputAxis = 0; outputAxis < axisMap.size(); ++outputAxis) {
         if (axisMap[outputAxis] >= 0 && static_cast<std::size_t>(axisMap[outputAxis]) == axis) {
            info.axis = static_cast<int>(outputAxis);
            return info;
         }
      }
      return std::nullopt;
   };

   auto isValidAxisMap = [](const std::vector<int_t> &axisMap, std::size_t sourceRank,
                            std::size_t targetRank) {
      if (axisMap.empty())
         return true;
      if (axisMap.size() != targetRank)
         return false;
      std::vector<bool> seen(sourceRank, false);
      for (auto axis : axisMap) {
         if (axis == -1)
            continue;
         if (axis < 0 || static_cast<std::size_t>(axis) >= sourceRank || seen[static_cast<std::size_t>(axis)])
            return false;
         seen[static_cast<std::size_t>(axis)] = true;
      }
      return true;
   };

   auto propagateSingleSourceMetadata = [&](const std::string &source,
                                            const std::string &target, const std::vector<int_t> &axisMap) {
      if (!HasQuantizationInfo(target) && HasQuantizationInfo(source)) {
         if (auto info = remapQuantization(GetQuantizationInfo(source), axisMap))
            AddQuantizationInfo(target, *info);
      }

      if (!HasLowPrecisionTensorInfo(target) && HasLowPrecisionTensorInfo(source)) {
         auto info = GetLowPrecisionTensorInfo(source);
         if (info.affineQuantization) {
            auto remapped = remapQuantization(*info.affineQuantization, axisMap);
            if (!remapped)
               return;
            info.affineQuantization = *remapped;
         }
         AddLowPrecisionTensorInfo(target, std::move(info));
      }
   };

   auto propagateCompatibleSourceMetadata = [&](const std::vector<std::string> &sources,
                                                const std::string &target, const std::vector<int_t> &axisMap) {
      if (sources.empty())
         return;

      if (!HasQuantizationInfo(target)) {
         bool compatible = true;
         std::optional<QuantizationInfo> candidate;
         for (const auto &source : sources) {
            if (!HasQuantizationInfo(source)) {
               compatible = false;
               continue;
            }
            const auto &info = GetQuantizationInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameQuantizationInfo(*candidate, info))
               compatible = false;
         }
         if (compatible && candidate) {
            if (auto remapped = remapQuantization(*candidate, axisMap))
               AddQuantizationInfo(target, *remapped);
         }
      }

      if (!HasLowPrecisionTensorInfo(target)) {
         bool compatible = true;
         std::optional<LowPrecisionTensorInfo> candidate;
         for (const auto &source : sources) {
            if (!HasLowPrecisionTensorInfo(source)) {
               compatible = false;
               continue;
            }
            const auto &info = GetLowPrecisionTensorInfo(source);
            if (!candidate)
               candidate = info;
            else if (!sameLowPrecisionTensorInfo(*candidate, info))
               compatible = false;
         }
         if (compatible && candidate) {
            if (candidate->affineQuantization) {
               auto remapped = remapQuantization(*candidate->affineQuantization, axisMap);
               if (!remapped)
                  return;
               candidate->affineQuantization = *remapped;
            }
            AddLowPrecisionTensorInfo(target, std::move(*candidate));
         }
      }
   };

   for (const auto &op : fOperators) {
      if (!op || !op->PropagatesQuantizationMetadata())
         continue;

      auto rawSources = op->GetQuantizationMetadataSourceTensors();
      std::vector<std::string> sources;
      sources.reserve(rawSources.size());
      for (const auto &rawSource : rawSources) {
         auto source = UTILITY::Clean_name(rawSource);
         if (!source.empty())
            sources.push_back(std::move(source));
      }
      if (sources.empty())
         continue;

      const auto &source = sources.front();
      auto sourceShape = GetTensorShape(source);
      for (const auto &rawTarget : op->GetQuantizationMetadataTargetTensors()) {
         const auto target = UTILITY::Clean_name(rawTarget);
         if (target.empty() || target == source)
            continue;
         auto targetShape = GetTensorShape(target);
         auto axisMap = op->GetQuantizationMetadataAxisMap(sourceShape, targetShape);
         if (!isValidAxisMap(axisMap, sourceShape.size(), targetShape.size()))
            continue;
         if (op->RequiresCompatibleQuantizationMetadataInputs())
            propagateCompatibleSourceMetadata(sources, target, axisMap);
         else
            propagateSingleSourceMetadata(source, target, axisMap);
      }
   }

   const auto graph = BuildQuantizationGraphIndex(fOperators);
   QuantizationPassContext context{*this, fOperators, QuantizationExtension::Of(*this).state, graph, fVerbose};
   // Each family exposes one Discover* entry that yields both regions and their
   // lowering plans; the model pass has no family-specific plan step.
   DiscoverQuantizedDenseLinearRegions(context);
   DiscoverQuantizedConvRegions(context);
   DiscoverQuantizedElementwiseRegions(context);
   DiscoverQuantizedGatherRegions(context);
}

} // namespace SOFIE
