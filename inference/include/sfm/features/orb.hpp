#pragma once
// ORB: oriented FAST keypoints on an image pyramid + steered BRIEF descriptors (Rublee et al.,
// ICCV 2011), following OpenCV's ORB in structure:
//   1. pyramid, ratio `scale`; level l gets a feature quota proportional to (1/scale)^l
//   2. FAST (threshold, border, mask) + 3x3 NMS per level; the 2 x quota best by FAST score are
//      re-ranked by the Harris response, the best `quota` are kept
//   3. orientation = intensity centroid of the radius-15 disc around the keypoint
//   4. descriptor = 256 intensity comparisons on the Gaussian-smoothed level, test pattern
//      rotated by the unit moment vector (m10, m01) / |m| (not via the angle: no atan2/cos/sin
//      round trip, so the CUDA version is bit-identical)
// Differences to OpenCV (so results are close, not identical): own 256-pair test pattern (seeded
// Gaussian, see rbrief_pattern), integer bilinear pyramid and blur, keypoints mapped to level 0
// with the pyramid's pixel-centre convention (x0 = (x + 0.5) * W0 / Wl - 0.5) instead of x * s^l.
// Ranking: FAST score, then the integer Harris response (exact::harris_int), ties by raster order.

#include <array>
#include <cstdint>
#include <vector>

#include "sfm/data/image_io.hpp"
#include "sfm/features/features.hpp"

namespace sfm::features {

struct OrbParams {
  std::uint32_t max_features = 2000;
  int levels = 8;
  double scale = 1.2;
  int fast_threshold = 10;
  int edge_border = 31;  // >= 22: the rotated test pattern (radius <= 13 * sqrt 2) and the
                         // orientation disc (radius 15) stay inside the level
};

// Quota per level: N (1 - f) / (1 - f^L) f^l with f = 1 / scale, rounded; the last level takes
// the remainder so the quotas sum to max_features.
std::vector<int> level_quotas(const OrbParams& p);

// Intensity-centroid angle in degrees [0, 360) of the radius-15 disc centred on (x, y), and the
// same from given moments.
float ic_angle(const data::Image8& img, int x, int y);
float angle_deg(std::int64_t m10, std::int64_t m01);

// The 256 test pairs (x1, y1, x2, y2), coordinates in [-13, 13], fixed for every build.
const std::array<std::array<std::int8_t, 4>, 256>& rbrief_pattern();

// Steered BRIEF on a smoothed image at integer position (x, y). Bit i = I(p1_i) < I(p2_i) with the
// pattern rotated by (c, s) = (cos, sin) of the orientation (rounded to whole pixels), or by the
// direction of the moments (m10, m01) (exact::steering).
void rbrief(const data::Image8& smoothed, int x, int y, double c, double s,
            std::uint32_t out[kDescriptorWords]);
void rbrief(const data::Image8& smoothed, int x, int y, std::int64_t m10, std::int64_t m01,
            std::uint32_t out[kDescriptorWords]);

// The whole detector. `mask` (optional, same size as gray, non-zero = allowed) restricts keypoint
// positions. Output order: by level, then by Harris response (descending), ties by raster order.
Features detect_orb(const data::Image8& gray, const data::Image8* mask, const OrbParams& p);

}  // namespace sfm::features
