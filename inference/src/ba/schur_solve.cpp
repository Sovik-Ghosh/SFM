// Schur complement on the camera block (exercise unit, see the P2 design).
//
// The damped normal equations   [U  W] [dc]   [-gc]
//                               [W' V] [dp] = [-gp]
// are block-eliminated: V is block diagonal (one 3x3 block per point), so V^-1 is cheap and
//   (U - W V^-1 W') dc = -gc + W V^-1 gp        (reduced camera system, size 6 nc)
//   dp = V^-1 (-gp - W' dc)                     (back-substitution, per point)
// The reduced system is small and dense: Cholesky (LDLT) solves it.
#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "sfm/ba/ba.hpp"

namespace sfm::ba {

bool schur_solve(const Problem& p, const NormalEquations& ne, double lambda,
                 std::vector<Eigen::Matrix<double, 6, 1>>& dc, std::vector<Eigen::Vector3d>& dp) {
  const std::size_t nc = p.cameras();
  const std::size_t np = p.points();

  // Free camera parameters get consecutive indices in the reduced system; fixed ones none.
  std::vector<std::array<int, 6>> col(nc);
  int n_free = 0;
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t k = 0; k < 6; ++k) col[c][k] = p.fixed[c][k] ? -1 : n_free++;
  }

  // V^-1 with Marquardt damping; a point with a singular block is left out (dp = 0).
  std::vector<Eigen::Matrix3d> Vinv(np);
  std::vector<bool> point_ok(np, true);
  for (std::size_t x = 0; x < np; ++x) {
    Eigen::Matrix3d V = ne.V[x];
    V.diagonal() += lambda * V.diagonal() + Eigen::Vector3d::Constant(1e-12);
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(V);
    point_ok[x] = lu.isInvertible();
    Vinv[x] = Eigen::Matrix3d::Zero();
    if (point_ok[x]) Vinv[x] = lu.inverse();
  }

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_free, n_free);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n_free);
  for (std::size_t c = 0; c < nc; ++c) {
    Eigen::Matrix<double, 6, 6> U = ne.U[c];
    U.diagonal() += lambda * U.diagonal() + Eigen::Matrix<double, 6, 1>::Constant(1e-12);
    for (std::size_t r = 0; r < 6; ++r) {
      if (col[c][r] < 0) continue;
      b(col[c][r]) -= ne.gc[c](static_cast<Eigen::Index>(r));
      for (std::size_t s = 0; s < 6; ++s) {
        if (col[c][s] >= 0) {
          S(col[c][r], col[c][s]) += U(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(s));
        }
      }
    }
  }

  // Each point couples every pair of cameras that observe it: S -= W_a V^-1 W_b^T for every
  // pair of its observations (a, b), and b += W_a V^-1 gp.
  std::vector<std::vector<std::size_t>> obs_of_point(np);
  for (std::size_t k = 0; k < p.obs.size(); ++k) {
    obs_of_point[static_cast<std::size_t>(p.obs[k].point)].push_back(k);
  }
  for (std::size_t x = 0; x < np; ++x) {
    if (!point_ok[x]) continue;
    const auto& ks = obs_of_point[x];
    for (const std::size_t ka : ks) {
      const auto ca = static_cast<std::size_t>(p.obs[ka].camera);
      const Eigen::Matrix<double, 6, 3> WV = ne.W[ka] * Vinv[x];
      const Eigen::Matrix<double, 6, 1> bv = WV * ne.gp[x];
      for (std::size_t r = 0; r < 6; ++r) {
        if (col[ca][r] >= 0) b(col[ca][r]) += bv(static_cast<Eigen::Index>(r));
      }
      for (const std::size_t kb : ks) {
        const auto cb = static_cast<std::size_t>(p.obs[kb].camera);
        const Eigen::Matrix<double, 6, 6> blk = WV * ne.W[kb].transpose();
        for (std::size_t r = 0; r < 6; ++r) {
          if (col[ca][r] < 0) continue;
          for (std::size_t s = 0; s < 6; ++s) {
            if (col[cb][s] >= 0) {
              S(col[ca][r], col[cb][s]) -=
                  blk(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(s));
            }
          }
        }
      }
    }
  }

  const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) return false;
  const Eigen::VectorXd sol = ldlt.solve(b);
  if (!sol.allFinite()) return false;

  dc.assign(nc, Eigen::Matrix<double, 6, 1>::Zero());
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t k = 0; k < 6; ++k) {
      if (col[c][k] >= 0) dc[c](static_cast<Eigen::Index>(k)) = sol(col[c][k]);
    }
  }
  dp.assign(np, Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector3d> rhs(np);
  for (std::size_t x = 0; x < np; ++x) rhs[x] = -ne.gp[x];
  for (std::size_t k = 0; k < p.obs.size(); ++k) {
    const auto c = static_cast<std::size_t>(p.obs[k].camera);
    rhs[static_cast<std::size_t>(p.obs[k].point)] -= ne.W[k].transpose() * dc[c];
  }
  for (std::size_t x = 0; x < np; ++x) {
    if (point_ok[x]) dp[x] = Vinv[x] * rhs[x];
  }
  return true;
}

}  // namespace sfm::ba
