#ifndef SOFIE_RWEIGHTFILE
#define SOFIE_RWEIGHTFILE

#include "SOFIE/SOFIE_common.hxx"

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SOFIE {

inline constexpr char kBinaryWeightFileMagic[8] = {'S', 'O', 'F', 'I', 'E', 'W', 'B', '1'};
inline constexpr std::uint32_t kBinaryWeightFileVersion = 1;
inline constexpr std::uint32_t kBinaryWeightFileEndianMarker = 0x01020304u;

using BinaryWeightTensorEntry = std::pair<std::string, const InitializedTensor *>;

bool IsBinaryWeightTensorType(ETensorType type);

std::vector<BinaryWeightTensorEntry>
CollectBinaryWeightTensors(const std::unordered_map<std::string, InitializedTensor> &tensors);

long WriteBinaryWeightFile(std::ostream &stream,
                           const std::unordered_map<std::string, InitializedTensor> &tensors);

template <class T>
void ReadBinaryWeightValue(std::istream &stream, T &value)
{
   static_assert(std::is_trivially_copyable_v<T>);
   stream.read(reinterpret_cast<char *>(&value), sizeof(T));
   if (!stream)
      throw std::runtime_error("sofie binary weight file is truncated");
}

inline void ReadBinaryWeightFileHeader(std::istream &stream, std::uint64_t expectedTensorCount)
{
   char magic[8]{};
   stream.read(magic, sizeof(magic));
   std::uint32_t version = 0;
   std::uint32_t endianMarker = 0;
   std::uint64_t tensorCount = 0;
   ReadBinaryWeightValue(stream, version);
   ReadBinaryWeightValue(stream, endianMarker);
   ReadBinaryWeightValue(stream, tensorCount);
   if (!std::equal(std::begin(magic), std::end(magic), std::begin(kBinaryWeightFileMagic)))
      throw std::runtime_error("sofie binary weight file has an invalid magic header");
   if (version != kBinaryWeightFileVersion)
      throw std::runtime_error("sofie binary weight file has an unsupported version");
   if (endianMarker != kBinaryWeightFileEndianMarker)
      throw std::runtime_error("sofie binary weight file uses an incompatible byte order");
   if (tensorCount != expectedTensorCount)
      throw std::runtime_error("sofie binary weight file tensor count does not match the generated model");
}

template <class T>
void ReadTensorFromBinaryStream(std::istream &stream, T &target,
                                const std::string &expectedName, ETensorType expectedType,
                                const std::vector<std::size_t> &expectedShape)
{
   std::uint32_t nameLength = 0;
   ReadBinaryWeightValue(stream, nameLength);
   std::string name(nameLength, 0);
   stream.read(name.data(), name.size());
   std::uint32_t type = 0;
   std::uint32_t rank = 0;
   ReadBinaryWeightValue(stream, type);
   ReadBinaryWeightValue(stream, rank);
   std::vector<std::size_t> shape(rank);
   for (auto &dim : shape) {
      std::uint64_t value = 0;
      ReadBinaryWeightValue(stream, value);
      dim = static_cast<std::size_t>(value);
   }
   std::uint64_t elementCount = 0;
   std::uint64_t byteCount = 0;
   ReadBinaryWeightValue(stream, elementCount);
   ReadBinaryWeightValue(stream, byteCount);
   std::size_t expectedElementCount = 1;
   for (auto dim : expectedShape)
      expectedElementCount *= dim;
   using Element = std::remove_cv_t<std::remove_reference_t<decltype(target[0])>>;
   if (name != expectedName || type != static_cast<std::uint32_t>(expectedType) || shape != expectedShape ||
       elementCount != expectedElementCount || byteCount != expectedElementCount * sizeof(Element))
      throw std::runtime_error("sofie binary weight tensor metadata does not match generated tensor " + expectedName);
   stream.read(reinterpret_cast<char *>(&target[0]), static_cast<std::streamsize>(byteCount));
   if (!stream)
      throw std::runtime_error("sofie binary weight data is truncated for tensor " + expectedName);
}

} // namespace SOFIE

#endif // SOFIE_RWEIGHTFILE
