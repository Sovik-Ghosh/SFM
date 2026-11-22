#include "sfm/eval/umeyama.hpp"

#include <string>

#include <Eigen/LU>
#include <Eigen/SVD>

#include "sfm/core/error.hpp"

namespace sfm::eval {

// Umeyama 1991, Theorem + Eqs. (40)-(42):
//   mu_s, mu_d       centroids;  var_s = mean ||src_i - mu_s||^2
//   Sigma            = mean (dst_i - mu_d)(src_i - mu_s)^T  = U D V^T  (SVD)
//   S                = I, or diag(1, 1, -1) when det U * det V < 0 (the best orthogonal fit would
//                      be a reflection; flip the direction of least variance instead)
//   R = U S V^T,  s = trace(D S) / var_s,  t = mu_d - s R mu_s
Similarity umeyama(const Eigen::Matrix3Xd& src, const Eigen::Matrix3Xd& dst) {
  const Eigen::Index n = src.cols();
  if (dst.cols() != n) {
    throw GeometryError("umeyama: src has " + std::to_string(n) + " points, dst has " +
                        std::to_string(dst.cols()));
  }
  if (n < 3) throw GeometryError("umeyama: need at least 3 points, got " + std::to_string(n));

  const Eigen::Vector3d mu_s = src.rowwise().mean();
  const Eigen::Vector3d mu_d = dst.rowwise().mean();
  const Eigen::Matrix3Xd xs = src.colwise() - mu_s;
  const Eigen::Matrix3Xd xd = dst.colwise() - mu_d;
  const double var_s = xs.squaredNorm() / static_cast<double>(n);
  const Eigen::Matrix3d sigma = xd * xs.transpose() / static_cast<double>(n);

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d& d = svd.singularValues();  // sorted, d(0) >= d(1) >= d(2) >= 0
  // Rank >= 2 is needed: with all points on one line the rotation about that line is free.
  // Coplanar points (rank 2) are fine; S then fixes the normal's direction.
  if (!(var_s > 0.0) || !(d(1) > 1e-12 * d(0))) {
    throw GeometryError("umeyama: points are collinear or coincide; rotation is not determined");
  }

  Eigen::Vector3d S = Eigen::Vector3d::Ones();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) S(2) = -1.0;

  Similarity out;
  out.R = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
  out.s = d.dot(S) / var_s;
  out.t = mu_d - out.s * out.R * mu_s;
  return out;
}

}  // namespace sfm::eval
