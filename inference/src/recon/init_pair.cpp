#include <algorithm>
#include <numeric>

#include "sfm/core/parallel.hpp"
#include "sfm/recon/incremental.hpp"
#include "sfm/recon/pnp.hpp"

namespace sfm::recon {

namespace {

// E = K_j^T F K_i inherits the noise of the 8-point F, and on narrow views a small change of F
// moves the pose a lot (bunny: the same pair qualified with one F and failed with another).
// Refine the pose on all correspondences: triangulate (camera i fixed), Gauss-Newton on camera j
// with the points fixed, renormalise ||t|| = 1; 10 rounds.
void refine_two_view(const Eigen::Matrix3d& Ki, const Eigen::Matrix3d& Kj,
                     const std::vector<Eigen::Vector2d>& uv1,
                     const std::vector<Eigen::Vector2d>& uv2,
                     const std::vector<Eigen::Vector2d>& x1, const std::vector<Eigen::Vector2d>& x2,
                     Pose& pj) {
  const Pose pi;
  for (int round = 0; round < 10; ++round) {
    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector2d> uv;
    const Mat34 P1 = pose_matrix(pi);
    const Mat34 P2 = pose_matrix(pj);
    for (std::size_t m = 0; m < x1.size(); ++m) {
      Eigen::Vector3d Xm;
      if (!triangulate_dlt({P1, P2}, {x1[m], x2[m]}, Xm)) continue;
      if (depth(pi, Xm) <= 0 || depth(pj, Xm) <= 0) continue;
      if (reprojection_error(Ki, pi, Xm, uv1[m]) > 8.0) continue;  // gross outliers out
      X.push_back(Xm);
      uv.push_back(uv2[m]);
    }
    if (X.size() < 8) return;
    Pose next = pj;
    refine_pose(Kj, X, uv, next, 5);
    const double n = next.t.norm();
    if (!(n > 0)) return;
    next.t /= n;
    pj = next;
  }
}

}  // namespace

namespace {

// A pair whose unrefined pose gives a median ray angle below this share of init_min_angle_deg
// is rejected before the (10-round) refinement (P9c). Measured on playground (23 872 pairs tried):
// refinement moves the median angle by at most 0.92 deg (median 0.014), so at 10 deg nothing
// that would qualify is dropped, and 99.8 % of the refinements are skipped.
constexpr double kPrecheckShare = 0.5;

// Median ray angle (deg) of the correspondences triangulated in front of both cameras within
// tri_max_reproj_px, camera i at the identity; `count` = their number (0 → median 0).
double median_ray_angle(const Eigen::Matrix3d& Ki, const Eigen::Matrix3d& Kj,
                        const std::vector<Eigen::Vector2d>& uv1,
                        const std::vector<Eigen::Vector2d>& uv2,
                        const std::vector<Eigen::Vector2d>& x1,
                        const std::vector<Eigen::Vector2d>& x2, const Pose& pj,
                        const IncrementalParams& p, int& count) {
  const Pose pi;  // identity
  const Mat34 P1 = pose_matrix(pi);
  const Mat34 P2 = pose_matrix(pj);
  const Eigen::Vector3d c1 = pi.centre();
  const Eigen::Vector3d c2 = pj.centre();
  std::vector<double> angles;
  for (std::size_t m = 0; m < x1.size(); ++m) {
    Eigen::Vector3d X;
    if (!triangulate_dlt({P1, P2}, {x1[m], x2[m]}, X)) continue;
    if (depth(pi, X) <= 0 || depth(pj, X) <= 0) continue;
    if (reprojection_error(Ki, pi, X, uv1[m]) > p.tri_max_reproj_px ||
        reprojection_error(Kj, pj, X, uv2[m]) > p.tri_max_reproj_px) {
      continue;
    }
    angles.push_back(ray_angle_deg(c1, c2, X));
  }
  count = static_cast<int>(angles.size());
  if (angles.empty()) return 0.0;
  std::nth_element(angles.begin(), angles.begin() + static_cast<std::ptrdiff_t>(angles.size() / 2),
                   angles.end());
  return angles[angles.size() / 2];
}

// The initial-pair test of one pair (see choose_initial_pair); false if it does not qualify.
bool try_pair(const matching::PairMatches& pm, const std::vector<features::Features>& features,
              const std::vector<Eigen::Matrix3d>& K, const IncrementalParams& p, InitialPair& out) {
  const auto i = static_cast<std::size_t>(pm.i);
  const auto j = static_cast<std::size_t>(pm.j);
  std::vector<Eigen::Vector2d> uv1;
  std::vector<Eigen::Vector2d> uv2;
  std::vector<Eigen::Vector2d> x1;
  std::vector<Eigen::Vector2d> x2;
  for (std::size_t m = 0; m < pm.size(); ++m) {
    const auto a = static_cast<std::size_t>(pm.idx[2 * m]);
    const auto b = static_cast<std::size_t>(pm.idx[2 * m + 1]);
    uv1.emplace_back(features[i].xy[2 * a], features[i].xy[2 * a + 1]);
    uv2.emplace_back(features[j].xy[2 * b], features[j].xy[2 * b + 1]);
    x1.push_back(normalise(K[i], uv1.back()));
    x2.push_back(normalise(K[j], uv2.back()));
  }
  const Eigen::Matrix3d E = essential_from_fundamental(pm.F, K[i], K[j]);
  int in_front = 0;
  Pose pj = select_relative_pose(E, x1, x2, in_front);
  int count = 0;
  const double before = median_ray_angle(K[i], K[j], uv1, uv2, x1, x2, pj, p, count);
  if (before < kPrecheckShare * p.init_min_angle_deg) return false;
  refine_two_view(K[i], K[j], uv1, uv2, x1, x2, pj);
  const double median = median_ray_angle(K[i], K[j], uv1, uv2, x1, x2, pj, p, count);
  if (count < p.init_min_inliers) return false;
  if (median < p.init_min_angle_deg) return false;
  out = {pm.i, pm.j, pj, count, median};
  return true;
}

}  // namespace

bool choose_initial_pair(const std::vector<matching::PairMatches>& pairs,
                         const std::vector<features::Features>& features,
                         const std::vector<Eigen::Matrix3d>& K, const IncrementalParams& p,
                         InitialPair& out) {
  std::vector<std::size_t> order;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    if (static_cast<int>(pairs[k].size()) >= p.init_min_inliers) order.push_back(k);
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return pairs[a].size() > pairs[b].size();  // input is in (i, j) order: stable keeps ties
  });
  // Candidates are tested in parallel batches; the first one in `order` that qualifies wins, as
  // in a sequential scan (video: hundreds of strong neighbour pairs are too narrow).
  const std::size_t batch = 4 * static_cast<std::size_t>(core::default_threads());
  for (std::size_t first = 0; first < order.size(); first += batch) {
    const std::size_t n = std::min(batch, order.size() - first);
    std::vector<InitialPair> found(n);
    std::vector<char> ok(n, 0);
    core::parallel_for(n, [&](std::size_t b) {
      ok[b] = try_pair(pairs[order[first + b]], features, K, p, found[b]) ? 1 : 0;
    });
    for (std::size_t b = 0; b < n; ++b) {
      if (ok[b] != 0) {
        out = found[b];
        return true;
      }
    }
  }
  return false;
}

}  // namespace sfm::recon
