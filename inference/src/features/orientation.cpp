// Intensity centroid (Rosin 1999): the vector from the keypoint to the centroid of the patch
// intensities gives a repeatable orientation. theta = atan2(m01, m10), m_pq = sum x^p y^q I(x, y).
#include <cmath>
#include <cstdint>

#include "sfm/features/exact.hpp"
#include "sfm/features/orb.hpp"

namespace sfm::features {

float angle_deg(std::int64_t m10, std::int64_t m01) {
  double deg = std::atan2(static_cast<double>(m01), static_cast<double>(m10)) * 180.0 / M_PI;
  if (deg < 0.0) deg += 360.0;
  if (deg >= 360.0) deg -= 360.0;
  return static_cast<float>(deg);
}

float ic_angle(const data::Image8& img, int x, int y) {
  std::int64_t m10 = 0;
  std::int64_t m01 = 0;
  exact::moments(img.pixels.data(), img.width, x, y, m10, m01);
  return angle_deg(m10, m01);
}

}  // namespace sfm::features
