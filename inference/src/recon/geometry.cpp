#include "sfm/recon/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

namespace sfm::recon {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

Eigen::Matrix3d nearest_essential(const Eigen::Matrix3d& M) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  return svd.matrixU() * Eigen::Vector3d(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
}

Eigen::Matrix3d essential_from_fundamental(const Eigen::Matrix3d& F, const Eigen::Matrix3d& K1,
                                           const Eigen::Matrix3d& K2) {
  return nearest_essential(K2.transpose() * F * K1);
}

std::array<Pose, 4> decompose_essential(const Eigen::Matrix3d& E) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  // E is defined up to sign, so U and V may be flipped to make them rotations.
  if (U.determinant() < 0) U = -U;
  if (V.determinant() < 0) V = -V;
  Eigen::Matrix3d W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Eigen::Matrix3d Ra = U * W * V.transpose();
  const Eigen::Matrix3d Rb = U * W.transpose() * V.transpose();
  const Eigen::Vector3d u3 = U.col(2);
  return {Pose{Ra, u3}, Pose{Ra, -u3}, Pose{Rb, u3}, Pose{Rb, -u3}};
}

bool triangulate_dlt(const std::vector<Mat34>& P, const std::vector<Eigen::Vector2d>& x,
                     Eigen::Vector3d& X) {
  const std::size_t n = P.size();
  if (n < 2 || x.size() != n) return false;
  Eigen::Matrix<double, Eigen::Dynamic, 4> A(static_cast<Eigen::Index>(2 * n), 4);
  for (std::size_t k = 0; k < n; ++k) {
    const auto r = static_cast<Eigen::Index>(2 * k);
    A.row(r) = x[k].x() * P[k].row(2) - P[k].row(0);
    A.row(r + 1) = x[k].y() * P[k].row(2) - P[k].row(1);
    // Rows have the scale of the projection; normalising them weights every view equally.
    A.row(r).normalize();
    A.row(r + 1).normalize();
  }
  const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 4>> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d h = svd.matrixV().col(3);
  if (std::fabs(h(3)) < 1e-12 * h.head<3>().norm()) return false;
  X = h.head<3>() / h(3);
  return true;
}

Pose select_relative_pose(const Eigen::Matrix3d& E, const std::vector<Eigen::Vector2d>& x1n,
                          const std::vector<Eigen::Vector2d>& x2n, int& in_front,
                          std::vector<std::uint8_t>* mask) {
  const auto candidates = decompose_essential(E);
  const Mat34 P1 = Mat34::Identity();
  int best = -1;
  std::size_t best_k = 0;
  std::vector<std::uint8_t> best_mask(x1n.size(), 0);
  for (std::size_t k = 0; k < candidates.size(); ++k) {
    const Pose& c = candidates[k];
    const Mat34 P2 = pose_matrix(c);
    std::vector<std::uint8_t> m(x1n.size(), 0);
    int count = 0;
    for (std::size_t i = 0; i < x1n.size(); ++i) {
      Eigen::Vector3d X;
      if (!triangulate_dlt({P1, P2}, {x1n[i], x2n[i]}, X)) continue;
      if (X.z() > 0 && depth(c, X) > 0) {
        m[i] = 1;
        ++count;
      }
    }
    if (count > best) {
      best = count;
      best_k = k;
      best_mask.swap(m);
    }
  }
  in_front = best;
  if (mask != nullptr) *mask = best_mask;
  return candidates[best_k];
}

Eigen::Vector2d project_px(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X) {
  const Eigen::Vector3d h = K * (pose.R * X + pose.t);
  return h.head<2>() / h.z();
}

double depth(const Pose& pose, const Eigen::Vector3d& X) { return (pose.R * X + pose.t).z(); }

double reprojection_error(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X,
                          const Eigen::Vector2d& uv) {
  return (project_px(K, pose, X) - uv).norm();
}

double ray_angle_deg(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2,
                     const Eigen::Vector3d& X) {
  const Eigen::Vector3d a = (X - c1).normalized();
  const Eigen::Vector3d b = (X - c2).normalized();
  return std::atan2(a.cross(b).norm(), a.dot(b)) * 180.0 / M_PI;
}

Eigen::Vector2d normalise(const Eigen::Matrix3d& K, const Eigen::Vector2d& uv) {
  // K is upper triangular: solve K y = (u, v, 1) by back-substitution.
  const double y1 = (uv.y() - K(1, 2)) / K(1, 1);
  const double y0 = (uv.x() - K(0, 2) - K(0, 1) * y1) / K(0, 0);
  return {y0, y1};
}

Mat34 pose_matrix(const Pose& pose) {
  Mat34 P;
  P << pose.R, pose.t;
  return P;
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
  const double theta = w.norm();
  const Eigen::Matrix3d W = skew(w);
  if (theta < 1e-8) return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
  return Eigen::Matrix3d::Identity() + std::sin(theta) / theta * W +
         (1 - std::cos(theta)) / (theta * theta) * W * W;
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

Eigen::Matrix3d nearest_rotation(const Eigen::Matrix3d& M) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1.0 : 1.0;
  return svd.matrixU() * D * svd.matrixV().transpose();
}

}  // namespace sfm::recon
