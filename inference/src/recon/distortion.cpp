#include "sfm/recon/distortion.hpp"

#include <Eigen/LU>

namespace sfm::recon {

namespace {

Eigen::Vector2d to_pixel(const Eigen::Matrix3d& K, const Eigen::Vector2d& x) {
  const Eigen::Vector3d p = K * Eigen::Vector3d(x.x(), x.y(), 1.0);
  return p.head<2>() / p.z();
}

Eigen::Vector2d to_normalised(const Eigen::Matrix3d& K, const Eigen::Vector2d& uv) {
  const Eigen::Vector3d x = K.inverse() * Eigen::Vector3d(uv.x(), uv.y(), 1.0);
  return x.head<2>() / x.z();
}

double factor(const Radial& d, const Eigen::Vector2d& x) {
  const double r2 = x.squaredNorm();
  return 1.0 + d.k1 * r2 + d.k2 * r2 * r2;
}

}  // namespace

Eigen::Vector2d distort(const Eigen::Matrix3d& K, const Radial& d, const Eigen::Vector2d& uv) {
  const Eigen::Vector2d x = to_normalised(K, uv);
  return to_pixel(K, x * factor(d, x));
}

Eigen::Vector2d undistort(const Eigen::Matrix3d& K, const Radial& d, const Eigen::Vector2d& uv) {
  const Eigen::Vector2d xd = to_normalised(K, uv);
  Eigen::Vector2d x = xd;
  for (int i = 0; i < 30; ++i) {
    const Eigen::Vector2d next = xd / factor(d, x);
    if ((next - x).squaredNorm() < 1e-24) {
      x = next;
      break;
    }
    x = next;
  }
  return to_pixel(K, x);
}

std::vector<features::Features> undistort_features(const std::vector<features::Features>& f,
                                                   const std::vector<Eigen::Matrix3d>& K,
                                                   const Radial& d) {
  std::vector<features::Features> out = f;
  if (!d.any()) return out;
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto& xy = out[i].xy;
    for (std::size_t k = 0; k + 1 < xy.size(); k += 2) {
      const Eigen::Vector2d u = undistort(K[i], d, {xy[k], xy[k + 1]});
      xy[k] = static_cast<float>(u.x());
      xy[k + 1] = static_cast<float>(u.y());
    }
  }
  return out;
}

}  // namespace sfm::recon
