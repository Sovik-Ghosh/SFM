// Steered BRIEF. The test pattern is BRIEF's "G II" sampling (Calonder et al. 2010): both points of
// a pair i.i.d. Gaussian around the centre with sigma = S / 5 for a patch of size S = 31, clipped
// to the patch. It is generated from a fixed splitmix64 seed, so it is identical on every build.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "sfm/features/exact.hpp"
#include "sfm/features/orb.hpp"

namespace sfm::features {

namespace {

std::uint64_t splitmix64(std::uint64_t& state) {
  std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::array<std::array<std::int8_t, 4>, 256> make_pattern() {
  constexpr double kSigma = 31.0 / 5.0;
  constexpr int kLimit = 13;
  std::uint64_t state = 0x5EED0F0BB1E5ULL;
  const auto uniform = [&] { return static_cast<double>(splitmix64(state) >> 11) * 0x1.0p-53; };
  const auto coord = [&] {
    // Box-Muller, one value per call; resample values outside the patch.
    for (;;) {
      const double u1 = 1.0 - uniform();
      const double u2 = uniform();
      const double g = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
      const long v = std::lround(g * kSigma);
      if (v >= -kLimit && v <= kLimit) return static_cast<std::int8_t>(v);
    }
  };
  std::array<std::array<std::int8_t, 4>, 256> pattern{};
  for (auto& pair : pattern) {
    do {
      pair = {coord(), coord(), coord(), coord()};
    } while (pair[0] == pair[2] && pair[1] == pair[3]);  // a point compared with itself
  }
  return pattern;
}

}  // namespace

const std::array<std::array<std::int8_t, 4>, 256>& rbrief_pattern() {
  static const auto kPattern = make_pattern();
  return kPattern;
}

void rbrief(const data::Image8& smoothed, int x, int y, double c, double s,
            std::uint32_t out[kDescriptorWords]) {
  const auto sample = [&](int px, int py) {
    int rx = 0;
    int ry = 0;
    exact::rotate_round(px, py, c, s, rx, ry);
    return smoothed
        .pixels[static_cast<std::size_t>(y + ry) * static_cast<std::size_t>(smoothed.width) +
                static_cast<std::size_t>(x + rx)];
  };
  std::fill(out, out + kDescriptorWords, 0U);
  const auto& pattern = rbrief_pattern();
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const auto& p = pattern[i];
    if (sample(p[0], p[1]) < sample(p[2], p[3])) out[i / 32] |= 1U << (i % 32);
  }
}

void rbrief(const data::Image8& smoothed, int x, int y, std::int64_t m10, std::int64_t m01,
            std::uint32_t out[kDescriptorWords]) {
  double c = 1.0;
  double s = 0.0;
  exact::steering(m10, m01, c, s);
  rbrief(smoothed, x, y, c, s, out);
}

}  // namespace sfm::features
