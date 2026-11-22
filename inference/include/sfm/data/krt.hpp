#pragma once
// Projection matrix <-> camera parameters.
//
//   P = lambda * K [R | t]      (3x4, defined only up to the non-zero scale lambda, sign included)
//
// K: upper-triangular intrinsics, normalised so K(2,2) = 1 and diag(K) > 0.
// R: rotation world -> camera (orthonormal, det R = +1).
// t: translation; the camera centre in world coordinates is C = -R^T t.
// With these three rules the decomposition is unique.

#include <Eigen/Core>

#include "sfm/data/calib.hpp"

namespace sfm::data {

struct Krt {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Eigen::Vector3d centre() const { return -R.transpose() * t; }
};

// K [R | t]  (lambda = 1).
Mat34 compose_projection(const Krt& krt);

// Inverse of compose_projection for any scale lambda != 0. Throws IoError if the left 3x3 block
// of P is singular (not a projective camera).
Krt decompose_projection(const Mat34& P);

}  // namespace sfm::data
