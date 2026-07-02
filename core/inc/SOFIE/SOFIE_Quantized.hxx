#ifndef SOFIE_QUANTIZED
#define SOFIE_QUANTIZED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace SOFIE {

// Portable quantized runtime helpers used by generated lowered quantized operators.

inline bool NearlyEqualQuantizedScale(double lhs, double rhs)
{
   const auto scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
   return std::fabs(lhs - rhs) <= (1.0e-12 * scale);
}

inline bool MakeQuantizedFixedPointMultiplier(double realMultiplier, std::int64_t &multiplier, int &shift)
{
   if (!std::isfinite(realMultiplier) || realMultiplier <= 0.0) {
      return false;
   }

   int exponent = 0;
   const double significand = std::frexp(realMultiplier, &exponent);
   auto q31 = static_cast<std::int64_t>(std::llround(significand * 2147483648.0));
   if (q31 == (std::int64_t{1} << 31)) {
      q31 /= 2;
      ++exponent;
   }

   const int candidateShift = 31 - exponent;
   if (q31 <= 0 || candidateShift < 0 || candidateShift >= 62) {
      return false;
   }

   multiplier = q31;
   shift = candidateShift;
   return true;
}

inline bool MakeExactIntegerScaleMultiplier(double scale, std::int64_t &multiplier, int &shift)
{
   if (!std::isfinite(scale) || scale < 0.0) {
      return false;
   }

   const auto rounded = std::llround(scale);
   if (!NearlyEqualQuantizedScale(scale, static_cast<double>(rounded))) {
      return false;
   }

   multiplier = rounded;
   shift = 0;
   return true;
}

enum class EQuantizedGemmActivation {
   None = 0,
   Relu = 1
};

struct QuantizedGemmParams {
   std::size_t m = 0;
   std::size_t n = 0;
   std::size_t k = 0;
   std::size_t tileN = 4;

   double scaleX = 1.0;
   double scaleW = 1.0;
   double scaleY = 1.0;
   double scaleB = 1.0;

   std::int32_t zeroX = 0;
   std::int32_t zeroW = 0;
   std::int32_t zeroY = 0;
   std::int32_t zeroB = 0;

   std::int32_t qminX = 0;
   std::int32_t qmaxX = 0;
   std::int32_t qminY = 0;
   std::int32_t qmaxY = 0;
   std::int32_t qminB = 0;
   std::int32_t qmaxB = 0;

   bool hasBias = false;

   bool useIntegerEpilogue = false;
   bool useAccumulatorBias = false;
   bool useIntegerBias = false;
   std::int64_t requantMultiplier = 0;
   int requantShift = 0;
   std::int64_t biasRequantMultiplier = 0;
   int biasRequantShift = 0;

   EQuantizedGemmActivation activation = EQuantizedGemmActivation::None;
};

namespace INTERNAL {

enum class EQuantizedGemmCPUBackend {
   Portable = 0,
   X86 = 1,
   ARM = 2
};

inline constexpr bool IsQuantizedGemmX86Target()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
   return true;
#else
   return false;
#endif
}

inline constexpr bool IsQuantizedGemmARMTarget()
{
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
   return true;
#else
   return false;
#endif
}

inline constexpr EQuantizedGemmCPUBackend SelectQuantizedGemmCPUBackend()
{
   // Selects the CPU backend family visible at compile time.
   // Portable remains the fallback for every target family.
   if constexpr (IsQuantizedGemmX86Target()) {
      return EQuantizedGemmCPUBackend::X86;
   } else if constexpr (IsQuantizedGemmARMTarget()) {
      return EQuantizedGemmCPUBackend::ARM;
   } else {
      return EQuantizedGemmCPUBackend::Portable;
   }
}

inline std::int32_t QuantizedGemmClamp(std::int32_t value, std::int32_t qmin, std::int32_t qmax)
{
   return std::max(qmin, std::min(qmax, value));
}

inline std::int32_t QuantizedGemmQuantizeClamp(double value, double scale, std::int32_t zero, std::int32_t qmin,
                                               std::int32_t qmax)
{
   auto quantized = static_cast<std::int32_t>(std::nearbyint((value / scale) + zero));
   return QuantizedGemmClamp(quantized, qmin, qmax);
}

inline std::int64_t QuantizedGemmRoundDivideByPowerOfTwo(std::int64_t value, int shift)
{
   if (shift <= 0) {
      const auto leftShift = -shift;
      if (leftShift >= 62) {
         return value >= 0 ? std::numeric_limits<std::int64_t>::max() : std::numeric_limits<std::int64_t>::min();
      }
      return value << leftShift;
   }

   if (shift >= 62) {
      return 0;
   }

   const std::uint64_t magnitude = value < 0 ? static_cast<std::uint64_t>(-value) : static_cast<std::uint64_t>(value);
   const std::uint64_t quotient = magnitude >> shift;
   const std::uint64_t remainder = magnitude & ((std::uint64_t{1} << shift) - 1);
   const std::uint64_t half = std::uint64_t{1} << (shift - 1);
   const bool roundUp = (remainder > half) || (remainder == half && (quotient & std::uint64_t{1}) != 0);
   const auto rounded = static_cast<std::int64_t>(quotient + (roundUp ? 1 : 0));
   return value < 0 ? -rounded : rounded;
}

inline std::int64_t QuantizedGemmApplyFixedPointMultiplier(std::int32_t accumulator, std::int64_t multiplier, int shift)
{
   return QuantizedGemmRoundDivideByPowerOfTwo(static_cast<std::int64_t>(accumulator) * multiplier, shift);
}

inline std::int32_t QuantizedGemmFinalizeY(std::int64_t yq, const QuantizedGemmParams &params)
{
   if (params.activation == EQuantizedGemmActivation::Relu) {
      yq = std::max<std::int64_t>(yq, params.zeroY);
   }
   yq = std::max<std::int64_t>(params.qminY, std::min<std::int64_t>(params.qmaxY, yq));
   return static_cast<std::int32_t>(yq);
}

inline float QuantizedGemmFinalizeAccumulator(std::int32_t accumulator, std::size_t col, const float *bias,
                                             const QuantizedGemmParams &params)
{
   const bool hasRuntimeBias = params.hasBias && bias != nullptr;
   const bool canUseIntegerEpilogue =
      params.useIntegerEpilogue && (!hasRuntimeBias || params.useAccumulatorBias || params.useIntegerBias);

   if (canUseIntegerEpilogue) {
      auto acc = accumulator;
      std::int32_t biasCentered = 0;
      if (hasRuntimeBias) {
         const auto bq = QuantizedGemmQuantizeClamp(static_cast<double>(bias[col]), params.scaleB, params.zeroB,
                                                    params.qminB, params.qmaxB);
         biasCentered = bq - params.zeroB;
         if (params.useAccumulatorBias) {
            acc += biasCentered;
         }
      }

      auto scaled = QuantizedGemmApplyFixedPointMultiplier(acc, params.requantMultiplier, params.requantShift);
      if (hasRuntimeBias && params.useIntegerBias && !params.useAccumulatorBias) {
         scaled += QuantizedGemmApplyFixedPointMultiplier(biasCentered, params.biasRequantMultiplier,
                                                          params.biasRequantShift);
      }
      const auto yq = QuantizedGemmFinalizeY(scaled + params.zeroY, params);
      return static_cast<float>(static_cast<double>(yq - params.zeroY) * params.scaleY);
   }

   double real = static_cast<double>(accumulator) * params.scaleX * params.scaleW;
   if (hasRuntimeBias) {
      const auto bq = QuantizedGemmQuantizeClamp(static_cast<double>(bias[col]), params.scaleB, params.zeroB,
                                                 params.qminB, params.qmaxB);
      real += static_cast<double>(bq - params.zeroB) * params.scaleB;
   }

   const auto yq = QuantizedGemmFinalizeY(
      static_cast<std::int64_t>(std::nearbyint((real / params.scaleY) + params.zeroY)), params);
   return static_cast<float>(static_cast<double>(yq - params.zeroY) * params.scaleY);
}

template <typename WeightT>
inline void QuantizedGemmCPU_Portable(float *output, const float *input, const WeightT *packedWeight, const float *bias,
                                      const QuantizedGemmParams &params)
{
   if (params.tileN == 0) {
      return;
   }

   thread_local std::vector<std::int32_t> xqScratch;
   xqScratch.resize(params.k);
   auto *xqRow = xqScratch.data();

   for (std::size_t row = 0; row < params.m; ++row) {
      for (std::size_t kk = 0; kk < params.k; ++kk) {
         xqRow[kk] = QuantizedGemmQuantizeClamp(static_cast<double>(input[row * params.k + kk]), params.scaleX,
                                                params.zeroX, params.qminX, params.qmaxX);
      }

      for (std::size_t colBase = 0; colBase < params.n; colBase += params.tileN) {
         const std::size_t block = colBase / params.tileN;
         const std::size_t columns = std::min<std::size_t>(params.tileN, params.n - colBase);

         if (params.tileN == 4) {
            std::int32_t acc[4] = {0, 0, 0, 0};

            for (std::size_t kk = 0; kk < params.k; ++kk) {
               const auto xCentered = xqRow[kk] - params.zeroX;
               const auto weightBase = (block * params.k + kk) * params.tileN;
               for (std::size_t ji = 0; ji < columns; ++ji) {
                  const auto wq = static_cast<std::int32_t>(packedWeight[weightBase + ji]);
                  acc[ji] += xCentered * (wq - params.zeroW);
               }
            }

            for (std::size_t ji = 0; ji < columns; ++ji) {
               const std::size_t col = colBase + ji;
               output[row * params.n + col] = QuantizedGemmFinalizeAccumulator(acc[ji], col, bias, params);
            }
         } else {
            std::vector<std::int32_t> acc(params.tileN);

            for (std::size_t kk = 0; kk < params.k; ++kk) {
               const auto xCentered = xqRow[kk] - params.zeroX;
               const auto weightBase = (block * params.k + kk) * params.tileN;
               for (std::size_t ji = 0; ji < columns; ++ji) {
                  const auto wq = static_cast<std::int32_t>(packedWeight[weightBase + ji]);
                  acc[ji] += xCentered * (wq - params.zeroW);
               }
            }

            for (std::size_t ji = 0; ji < columns; ++ji) {
               const std::size_t col = colBase + ji;
               output[row * params.n + col] = QuantizedGemmFinalizeAccumulator(acc[ji], col, bias, params);
            }
         }
      }
   }
}

template <typename WeightT>
inline void QuantizedGemmCPU_Dispatch(float *output, const float *input, const WeightT *packedWeight, const float *bias,
                                      const QuantizedGemmParams &params)
{
   [[maybe_unused]] constexpr auto backend = SelectQuantizedGemmCPUBackend();
   // The dispatch boundary is explicit even when the portable kernel is selected.
   // Architecture-specific kernels can replace this call without changing generated code.
   QuantizedGemmCPU_Portable(output, input, packedWeight, bias, params);
}

} // namespace INTERNAL

template <typename WeightT>
inline void QuantizedGemm_Call(float *output, const float *input, const WeightT *packedWeight, const float *bias,
                               const QuantizedGemmParams &params)
{
   static_assert(std::is_same_v<WeightT, std::int8_t> || std::is_same_v<WeightT, std::uint8_t>,
                 "SOFIE::QuantizedGemm_Call supports int8/uint8 packed weights");

   INTERNAL::QuantizedGemmCPU_Dispatch(output, input, packedWeight, bias, params);
}

} // namespace SOFIE

#endif // SOFIE_QUANTIZED
