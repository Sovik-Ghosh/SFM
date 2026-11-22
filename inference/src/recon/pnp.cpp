#include "sfm/recon/pnp.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "sfm/eval/rng.hpp"

namespace sfm::recon {

std::vector<Pose> p3p(const std::array<Eigen::Vector3d, 3>& X,
                      const std::array<Eigen::Vector3d, 3>& bearing) {
  std::vector<Pose> out;
  const Eigen::Vector3d j1 = bearing[0].normalized();
  const Eigen::Vector3d j2 = bearing[1].normalized();
  const Eigen::Vector3d j3 = bearing[2].normalized();
  // Side lengths of the world triangle and cosines of the angles between the rays.
  const double a2 = (X[1] - X[2]).squaredNorm();
  const double b2 = (X[0] - X[2]).squaredNorm();
  const double c2 = (X[0] - X[1]).squaredNorm();
  if (!(a2 > 1e-24 && b2 > 1e-24 && c2 > 1e-24)) return out;
  if ((X[1] - X[0]).cross(X[2] - X[0]).squaredNorm() < 1e-18 * a2 * b2) return out;  // collinear
  const double ca = j2.dot(j3);
  const double cb = j1.dot(j3);
  const double cg = j1.dot(j2);

  // With s2 = u s1, s3 = v s1 (distances along the rays), the law of cosines in the three
  // triangles gives a quartic in v (Haralick et al. 1994, eq. 8 / 9).
  const double amc = (a2 - c2) / b2;
  const double apc = (a2 + c2) / b2;
  const double bmc = (b2 - c2) / b2;
  const double bma = (b2 - a2) / b2;
  const double A4 = (amc - 1) * (amc - 1) - 4 * c2 / b2 * ca * ca;
  const double A3 = 4 * (amc * (1 - amc) * cb - (1 - apc) * ca * cg + 2 * c2 / b2 * ca * ca * cb);
  const double A2 = 2 * (amc * amc - 1 + 2 * amc * amc * cb * cb + 2 * bmc * ca * ca -
                         4 * apc * ca * cb * cg + 2 * bma * cg * cg);
  const double A1 = 4 * (-amc * (1 + amc) * cb + 2 * a2 / b2 * cg * cg * cb - (1 - apc) * ca * cg);
  const double A0 = (1 + amc) * (1 + amc) - 4 * a2 / b2 * cg * cg;
  if (std::fabs(A4) < 1e-15) return out;

  // Real roots of the quartic: eigenvalues of its companion matrix, then two Newton steps.
  Eigen::Matrix4d C = Eigen::Matrix4d::Zero();
  C(0, 0) = -A3 / A4;
  C(0, 1) = -A2 / A4;
  C(0, 2) = -A1 / A4;
  C(0, 3) = -A0 / A4;
  C(1, 0) = 1;
  C(2, 1) = 1;
  C(3, 2) = 1;
  const Eigen::EigenSolver<Eigen::Matrix4d> es(C, false);
  for (Eigen::Index k = 0; k < 4; ++k) {
    const std::complex<double> root = es.eigenvalues()(k);
    if (std::fabs(root.imag()) > 1e-6 * std::max(1.0, std::fabs(root.real()))) continue;
    double v = root.real();
    for (int it = 0; it < 2; ++it) {
      const double f = (((A4 * v + A3) * v + A2) * v + A1) * v + A0;
      const double df = ((4 * A4 * v + 3 * A3) * v + 2 * A2) * v + A1;
      if (std::fabs(df) > 1e-300) v -= f / df;
    }
    if (v <= 0) continue;
    const double den = 2 * (cg - v * ca);
    if (std::fabs(den) < 1e-12) continue;
    const double u = ((-1 + amc) * v * v - 2 * amc * cb * v + 1 + amc) / den;
    if (u <= 0) continue;
    const double s1sq = c2 / (1 + u * u - 2 * u * cg);
    if (!(s1sq > 0)) continue;
    const double s1 = std::sqrt(s1sq);
    out.push_back(absolute_orientation({X[0], X[1], X[2]}, {s1 * j1, u * s1 * j2, v * s1 * j3}));
  }
  return out;
}

Pose absolute_orientation(const std::vector<Eigen::Vector3d>& p,
                          const std::vector<Eigen::Vector3d>& q) {
  Eigen::Vector3d cp = Eigen::Vector3d::Zero();
  Eigen::Vector3d cq = Eigen::Vector3d::Zero();
  for (std::size_t k = 0; k < p.size(); ++k) {
    cp += p[k];
    cq += q[k];
  }
  cp /= static_cast<double>(p.size());
  cq /= static_cast<double>(p.size());
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (std::size_t k = 0; k < p.size(); ++k) H += (q[k] - cq) * (p[k] - cp).transpose();
  const Eigen::Matrix3d R = nearest_rotation(H);  // argmax trace(R^T H)
  return {R, cq - R * cp};
}

double refine_pose(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                   const std::vector<Eigen::Vector2d>& uv, Pose& pose, int iterations) {
  const auto sq_error = [&](const Pose& ps) {
    double e = 0.0;
    for (std::size_t k = 0; k < X.size(); ++k) e += (project_px(K, ps, X[k]) - uv[k]).squaredNorm();
    return e;
  };
  double err = sq_error(pose);
  for (int it = 0; it < iterations; ++it) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (std::size_t k = 0; k < X.size(); ++k) {
      const Eigen::Vector3d RX = pose.R * X[k];
      const Eigen::Vector3d xc = RX + pose.t;
      if (xc.z() <= 0) continue;
      const double iz = 1.0 / xc.z();
      Eigen::Matrix<double, 2, 3> Jp;
      Jp << K(0, 0) * iz, K(0, 1) * iz, -(K(0, 0) * xc.x() + K(0, 1) * xc.y()) * iz * iz, 0,
          K(1, 1) * iz, -K(1, 1) * xc.y() * iz * iz;
      Eigen::Matrix<double, 2, 6> J;
      J.leftCols<3>() = -Jp * skew(RX);
      J.rightCols<3>() = Jp;
      const Eigen::Vector2d r = project_px(K, pose, X[k]) - uv[k];
      H += J.transpose() * J;
      g += J.transpose() * r;
    }
    const Eigen::Matrix<double, 6, 1> d = H.ldlt().solve(-g);
    if (!d.allFinite()) break;
    Pose next{exp_so3(d.head<3>()) * pose.R, pose.t + d.tail<3>()};
    const double e = sq_error(next);
    if (!(e < err)) break;  // Gauss-Newton without damping: stop at the first non-decrease
    pose = next;
    const bool small = err - e < 1e-12 * err;
    err = e;
    if (small) break;
  }
  return X.empty() ? 0.0 : std::sqrt(err / static_cast<double>(X.size()));
}

namespace {

// MSAC score: sum of min(e^2, thr^2) over all correspondences (behind the camera: thr^2).
double msac_cost(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                 const std::vector<Eigen::Vector2d>& uv, const Pose& pose, double thr2) {
  double c = 0.0;
  for (std::size_t k = 0; k < X.size(); ++k) {
    const double e =
        depth(pose, X[k]) > 0 ? (project_px(K, pose, X[k]) - uv[k]).squaredNorm() : thr2;
    c += std::min(e, thr2);
  }
  return c;
}

int count_inliers(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                  const std::vector<Eigen::Vector2d>& uv, const Pose& pose, double thr2,
                  std::vector<std::uint8_t>* mask) {
  int n = 0;
  for (std::size_t k = 0; k < X.size(); ++k) {
    const bool in =
        depth(pose, X[k]) > 0 && (project_px(K, pose, X[k]) - uv[k]).squaredNorm() < thr2;
    if (mask != nullptr) (*mask)[k] = in ? 1 : 0;
    n += in ? 1 : 0;
  }
  return n;
}

}  // namespace

PnpResult pnp_ransac(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                     const std::vector<Eigen::Vector2d>& uv, const PnpParams& p) {
  PnpResult r;
  const std::size_t n = X.size();
  r.inlier.assign(n, 0);
  if (n < 4 || uv.size() != n) return r;
  std::vector<Eigen::Vector3d> bearing(n);
  for (std::size_t k = 0; k < n; ++k) bearing[k] = normalise(K, uv[k]).homogeneous().normalized();

  const double thr2 = p.threshold_px * p.threshold_px;
  eval::Rng rng(p.seed);
  std::vector<std::size_t> perm(n);
  double best_cost = std::numeric_limits<double>::infinity();
  int best = -1;
  Pose best_pose;
  int limit = p.max_iters;
  int it = 0;
  for (; it < limit; ++it) {
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    std::array<Eigen::Vector3d, 3> sX;
    std::array<Eigen::Vector3d, 3> sb;
    for (std::size_t k = 0; k < 3; ++k) {
      std::swap(perm[k], perm[k + rng.index(n - k)]);
      sX[k] = X[perm[k]];
      sb[k] = bearing[perm[k]];
    }
    for (const Pose& pose : p3p(sX, sb)) {
      const double cost = msac_cost(K, X, uv, pose, thr2);
      if (!(cost < best_cost)) continue;
      best_cost = cost;
      const int c = count_inliers(K, X, uv, pose, thr2, nullptr);
      best = c;
      best_pose = pose;
      const double w3 = std::pow(static_cast<double>(c) / static_cast<double>(n), 3);
      if (w3 >= 1.0) {
        limit = it + 1;
      } else if (w3 > 0.0) {
        const double need = std::log(1.0 - p.confidence) / std::log(1.0 - w3);
        if (need < limit) limit = std::max(it + 1, static_cast<int>(std::ceil(need)));
      }
    }
  }
  r.iterations = it;
  if (best < 4) return r;

  // Refine on the inliers and update them; twice (the set can grow after refinement).
  Pose pose = best_pose;
  std::vector<std::uint8_t> mask(n, 0);
  count_inliers(K, X, uv, pose, thr2, &mask);
  for (int round = 0; round < 2; ++round) {
    std::vector<Eigen::Vector3d> iX;
    std::vector<Eigen::Vector2d> iuv;
    for (std::size_t k = 0; k < n; ++k) {
      if (mask[k] != 0) {
        iX.push_back(X[k]);
        iuv.push_back(uv[k]);
      }
    }
    if (iX.size() < 4) break;
    refine_pose(K, iX, iuv, pose);
    count_inliers(K, X, uv, pose, thr2, &mask);
  }
  r.pose = pose;
  r.inlier = mask;
  r.inliers = static_cast<int>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
  r.ok = r.inliers >= p.min_inliers &&
         static_cast<double>(r.inliers) >= p.min_inlier_ratio * static_cast<double>(n);
  return r;
}

}  // namespace sfm::recon
