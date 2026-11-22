// Normalised 8-point algorithm. Exercise unit (see the P2 design): the comments give the maths.
#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "sfm/core/error.hpp"
#include "sfm/matching/fundamental.hpp"

namespace sfm::matching {

Eigen::Matrix3d hartley_normalisation(const Points2& x) {
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  for (const auto& p : x) c += p;
  c /= static_cast<double>(x.size());
  double mean_dist = 0.0;
  for (const auto& p : x) mean_dist += (p - c).norm();
  mean_dist /= static_cast<double>(x.size());
  if (!(mean_dist > 1e-12)) throw GeometryError("hartley_normalisation: all points coincide");
  const double s = std::sqrt(2.0) / mean_dist;
  Eigen::Matrix3d T;
  T << s, 0, -s * c.x(), 0, s, -s * c.y(), 0, 0, 1;
  return T;
}

Eigen::Matrix3d eight_point(const Points2& x1, const Points2& x2) {
  if (x1.size() != x2.size()) throw GeometryError("eight_point: point counts differ");
  if (x1.size() < 8) throw GeometryError("eight_point: needs >= 8 correspondences");
  const Eigen::Matrix3d T1 = hartley_normalisation(x1);
  const Eigen::Matrix3d T2 = hartley_normalisation(x2);

  // x2^T F x1 = 0 is linear in f = (F00, F01, F02, F10, ..., F22):
  //   [u2 u1, u2 v1, u2, v2 u1, v2 v1, v2, u1, v1, 1] . f = 0
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(static_cast<Eigen::Index>(x1.size()), 9);
  for (std::size_t k = 0; k < x1.size(); ++k) {
    const Eigen::Vector3d a = T1 * x1[k].homogeneous();
    const Eigen::Vector3d b = T2 * x2[k].homogeneous();
    A.row(static_cast<Eigen::Index>(k)) << b.x() * a.x(), b.x() * a.y(), b.x(), b.y() * a.x(),
        b.y() * a.y(), b.y(), a.x(), a.y(), 1.0;
  }
  // Least-squares f with ||f|| = 1: right singular vector of the smallest singular value.
  const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd_a(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 1> f = svd_a.matrixV().col(8);
  Eigen::Matrix3d Fn;
  Fn << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

  // Rank 2 (every F has a null space, the epipole): the nearest rank-2 matrix in Frobenius norm
  // drops the smallest singular value.
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd_f(Fn, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svd_f.singularValues();
  s(2) = 0.0;
  Fn = svd_f.matrixU() * s.asDiagonal() * svd_f.matrixV().transpose();

  // Back to pixels: x2n^T Fn x1n = x2^T (T2^T Fn T1) x1.
  Eigen::Matrix3d F = T2.transpose() * Fn * T1;
  const double norm = F.norm();
  if (!(norm > 0.0)) throw GeometryError("eight_point: degenerate configuration");
  F /= norm;
  // Fix the sign so equal inputs give bit-equal outputs across SVD sign choices.
  Eigen::Index r = 0;
  Eigen::Index c = 0;
  F.cwiseAbs().maxCoeff(&r, &c);
  if (F(r, c) < 0) F = -F;
  return F;
}

double sampson_sq(const Eigen::Matrix3d& F, const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) {
  const Eigen::Vector3d a = x1.homogeneous();
  const Eigen::Vector3d b = x2.homogeneous();
  const Eigen::Vector3d Fa = F * a;
  const Eigen::Vector3d Ftb = F.transpose() * b;
  const double e = b.dot(Fa);
  const double denom = Fa(0) * Fa(0) + Fa(1) * Fa(1) + Ftb(0) * Ftb(0) + Ftb(1) * Ftb(1);
  return denom > 0.0 ? e * e / denom : 0.0;
}

}  // namespace sfm::matching
