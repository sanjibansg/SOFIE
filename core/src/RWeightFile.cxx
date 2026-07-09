#include "SOFIE/RWeightFile.hxx"

#include <algorithm>
#include <stdexcept>

namespace SOFIE {

bool IsBinaryWeightTensorType(ETensorType type)
{
   return type == ETensorType::FLOAT || type == ETensorType::INT8 || type == ETensorType::UINT8 ||
          type == ETensorType::INT32 || type == ETensorType::INT64;
}

std::vector<BinaryWeightTensorEntry>
CollectBinaryWeightTensors(const std::unordered_map<std::string, InitializedTensor> &tensors)
{
   std::vector<BinaryWeightTensorEntry> result;
   result.reserve(tensors.size());
   for (const auto &[name, tensor] : tensors) {
      if (tensor.IsWeightTensor())
         result.emplace_back(name, &tensor);
   }
   std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
   return result;
}

long WriteBinaryWeightFile(std::ostream &stream,
                           const std::unordered_map<std::string, InitializedTensor> &tensors)
{
   auto writeValue = [&stream](const auto &value) {
      stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
   };
   const auto weights = CollectBinaryWeightTensors(tensors);
   stream.write(kBinaryWeightFileMagic, sizeof(kBinaryWeightFileMagic));
   writeValue(kBinaryWeightFileVersion);
   writeValue(kBinaryWeightFileEndianMarker);
   writeValue(static_cast<std::uint64_t>(weights.size()));

   for (const auto &[sourceName, tensor] : weights) {
      const auto type = tensor->type();
      if (!IsBinaryWeightTensorType(type))
         throw std::runtime_error("sofie tensor " + sourceName + " has a type unsupported by binary weights");
      const std::string name = "tensor_" + sourceName;
      const auto nameLength = static_cast<std::uint32_t>(name.size());
      const auto typeCode = static_cast<std::uint32_t>(type);
      const auto rank = static_cast<std::uint32_t>(tensor->shape().size());
      const auto elementCount = static_cast<std::uint64_t>(ConvertShapeToLength(tensor->shape()));
      const auto byteCount = elementCount * GetTypeSize(type);
      writeValue(nameLength);
      stream.write(name.data(), name.size());
      writeValue(typeCode);
      writeValue(rank);
      for (auto dim : tensor->shape())
         writeValue(static_cast<std::uint64_t>(dim));
      writeValue(elementCount);
      writeValue(byteCount);
      stream.write(static_cast<const char *>(tensor->sharedptr().get()), static_cast<std::streamsize>(byteCount));
      if (!stream)
         throw std::runtime_error("sofie failed to write binary tensor " + name);
   }
   return static_cast<long>(stream.tellp());
}

} // namespace SOFIE
