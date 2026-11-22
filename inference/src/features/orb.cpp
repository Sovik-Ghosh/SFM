#include "sfm/features/orb.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "sfm/features/exact.hpp"
#include "sfm/features/fast.hpp"
#include "sfm/features/pyramid.hpp"

namespace sfm::features {

std::vector<int> level_quotas(const OrbParams& p) {
  const double f = 1.0 / p.scale;
  const auto n = static_cast<double>(p.max_features);
  double per = n * (1.0 - f) / (1.0 - std::pow(f, p.levels));
  std::vector<int> q(static_cast<std::size_t>(p.levels), 0);
  int sum = 0;
  for (int l = 0; l + 1 < p.levels; ++l) {
    q[static_cast<std::size_t>(l)] = static_cast<int>(std::lround(per));
    sum += q[static_cast<std::size_t>(l)];
    per *= f;
  }
  q.back() = std::max(0, static_cast<int>(p.max_features) - sum);
  return q;
}

Features detect_orb(const data::Image8& gray, const data::Image8* mask, const OrbParams& p) {
  if (p.edge_border < 22) throw std::invalid_argument("detect_orb: edge_border must be >= 22");
  if (mask != nullptr && (mask->width != gray.width || mask->height != gray.height)) {
    throw std::invalid_argument("detect_orb: mask size differs from the image");
  }
  const auto levels = build_pyramid(gray, p.levels, p.scale, Resample::kBilinear);
  std::vector<data::Image8> masks;
  if (mask != nullptr) masks = build_pyramid(*mask, p.levels, p.scale, Resample::kNearest);
  const auto quotas = level_quotas(p);

  Features out;
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const data::Image8& img = levels[l];
    const int quota = quotas[l];
    if (quota <= 0 || img.width <= 2 * p.edge_border || img.height <= 2 * p.edge_border) continue;

    auto corners =
        detect_fast(img, mask != nullptr ? &masks[l] : nullptr, p.fast_threshold, p.edge_border);
    // Raster order in, so stable_sort keeps raster order among equal scores.
    std::stable_sort(corners.begin(), corners.end(),
                     [](const Corner& a, const Corner& b) { return a.score > b.score; });
    corners.resize(std::min(corners.size(), static_cast<std::size_t>(2 * quota)));

    struct Ranked {
      Corner c;
      std::int64_t harris = 0;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(corners.size());
    for (const Corner& c : corners) ranked.push_back({c, harris_int(img, c.x, c.y)});
    std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
      if (a.harris != b.harris) return a.harris > b.harris;
      return a.c.y != b.c.y ? a.c.y < b.c.y : a.c.x < b.c.x;
    });
    ranked.resize(std::min(ranked.size(), static_cast<std::size_t>(quota)));

    const data::Image8 smooth = gaussian_blur7(img);
    const double sx = static_cast<double>(gray.width) / img.width;
    const double sy = static_cast<double>(gray.height) / img.height;
    for (const Ranked& r : ranked) {
      std::int64_t m10 = 0;
      std::int64_t m01 = 0;
      exact::moments(img.pixels.data(), img.width, r.c.x, r.c.y, m10, m01);
      const float angle = angle_deg(m10, m01);
      std::uint32_t d[kDescriptorWords];
      rbrief(smooth, r.c.x, r.c.y, m10, m01, d);
      out.xy.push_back(static_cast<float>((r.c.x + 0.5) * sx - 0.5));
      out.xy.push_back(static_cast<float>((r.c.y + 0.5) * sy - 0.5));
      out.angle.push_back(angle);
      out.response.push_back(exact::harris_response(r.harris));
      out.octave.push_back(static_cast<std::uint8_t>(l));
      out.desc.insert(out.desc.end(), d, d + kDescriptorWords);
    }
  }
  return out;
}

}  // namespace sfm::features
