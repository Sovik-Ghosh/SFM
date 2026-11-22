#pragma once
// Exact two-view geometry for tests: F from two projection matrices.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace sfm::matching::test {

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// F with x2^T F x1 = 0: F = [e2]_x P2 P1^+, e2 = P2 C1, C1 = null space of P1. Unit Frobenius norm.
inline Eigen::Matrix3d fundamental_from_projections(const Eigen::Matrix<double, 3, 4>& P1,
                                                    const Eigen::Matrix<double, 3, 4>& P2) {
  const Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(P1, Eigen::ComputeFullV);
  const Eigen::Vector4d C1 = svd.matrixV().col(3);
  const Eigen::Matrix<double, 4, 3> pinv = P1.transpose() * (P1 * P1.transpose()).inverse();
  Eigen::Matrix3d F = skew(P2 * C1) * P2 * pinv;
  return F / F.norm();
}

// Scale-and-sign-free distance between two F estimates.
inline double f_distance(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  const Eigen::Matrix3d an = a / a.norm();
  const Eigen::Matrix3d bn = b / b.norm();
  return std::min((an - bn).norm(), (an + bn).norm());
}

}  // namespace sfm::matching::test
