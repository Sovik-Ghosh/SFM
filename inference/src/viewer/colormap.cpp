#include "sfm/viewer/colormap.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sfm/viewer/colormap_hd.hpp"

namespace sfm::viewer {

Rgb viridis(float v) {
  Rgb out{};
  hd::viridis(v, out.data());  // shared with the CUDA interop kernel (byte-identical)
  return out;
}

std::array<float, 2> robust_range(std::vector<float> v, float q_lo, float q_hi) {
  if (v.empty()) return {0.0F, 1.0F};
  std::sort(v.begin(), v.end());
  const auto at = [&](float q) {
    const auto k = static_cast<std::size_t>(std::lround(q * static_cast<float>(v.size() - 1)));
    return v[std::min(k, v.size() - 1)];
  };
  const float lo = at(q_lo);
  float hi = at(q_hi);
  if (!(hi > lo)) hi = lo + 1.0F;
  return {lo, hi};
}

float normalise(float v, float lo, float hi) { return hd::normalise(v, lo, hi); }

}  // namespace sfm::viewer
