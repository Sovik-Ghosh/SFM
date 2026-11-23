#pragma once
// Radial lens distortion, as COLMAP's OPENCV / RADIAL camera models (tangential terms 0):
//   x = K^-1 (u, v, 1) (normalised, undistorted), r^2 = x^2 + y^2,
//   x_d = x (1 + k1 r^2 + k2 r^4)          (what the lens records)
// The pipeline works on undistorted keypoints (pinhole); exports keep the recorded pixels.

#include <vector>

#include <Eigen/Core>

#include "sfm/features/features.hpp"

namespace sfm::recon {

struct Radial {
  double k1 = 0.0;
  double k2 = 0.0;
  bool any() const { return k1 != 0.0 || k2 != 0.0; }
};

// Ideal pixel -> recorded pixel.
Eigen::Vector2d distort(const Eigen::Matrix3d& K, const Radial& d, const Eigen::Vector2d& uv);

// Recorded pixel -> ideal pixel (fixed-point iteration, converges for |k1 r^2| < 0.5).
Eigen::Vector2d undistort(const Eigen::Matrix3d& K, const Radial& d, const Eigen::Vector2d& uv);

// Copies of the features with every keypoint undistorted (K[i] for image i).
std::vector<features::Features> undistort_features(const std::vector<features::Features>& f,
                                                   const std::vector<Eigen::Matrix3d>& K,
                                                   const Radial& d);

}  // namespace sfm::recon
