#pragma once
// Least-squares similarity between two point sets (Umeyama 1991, "Least-squares estimation of
// transformation parameters between two point patterns", IEEE PAMI 13(4)).
//
//   dst_i ~ s * R * src_i + t      minimising  sum_i || dst_i - (s R src_i + t) ||^2
//
// R is a proper rotation (det +1) even when the best orthogonal fit would be a reflection.
// Used to compare a reconstruction (arbitrary frame and scale) with ground truth.

#include <Eigen/Core>

namespace sfm::eval {

struct Similarity {
  double s = 1.0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Eigen::Vector3d apply(const Eigen::Vector3d& x) const { return s * R * x + t; }
};

// src and dst: 3 x N, column i of src corresponds to column i of dst. Throws sfm::GeometryError if
// the sizes differ, N < 3, or the points do not determine a rotation (all collinear or identical).
Similarity umeyama(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& dst);

}  // namespace sfm::eval
