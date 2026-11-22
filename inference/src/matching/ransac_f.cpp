#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

#include "sfm/core/error.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/matching/fundamental.hpp"

namespace sfm::matching {

namespace {

struct Score {
  int inliers = 0;
  double error = std::numeric_limits<double>::infinity();

  bool better_than(const Score& o) const {
    return inliers > o.inliers || (inliers == o.inliers && error < o.error);
  }
};

Score evaluate(const Eigen::Matrix3d& F, const Points2& x1, const Points2& x2, double thr2,
               std::vector<std::uint8_t>* inlier) {
  Score s;
  s.error = 0.0;
  for (std::size_t k = 0; k < x1.size(); ++k) {
    const double e = sampson_sq(F, x1[k], x2[k]);
    const bool in = e < thr2;
    if (inlier != nullptr) (*inlier)[k] = in ? 1 : 0;
    if (in) {
      ++s.inliers;
      s.error += e;
    }
  }
  return s;
}

int needed_iterations(double inlier_share, double confidence, int max_iters) {
  const double w8 = std::pow(inlier_share, 8);
  if (w8 <= 0.0) return max_iters;
  if (w8 >= 1.0) return 1;
  const double n = std::log(1.0 - confidence) / std::log(1.0 - w8);
  return n >= max_iters ? max_iters : std::max(1, static_cast<int>(std::ceil(n)));
}

}  // namespace

RansacResult ransac_fundamental(const Points2& x1, const Points2& x2, const RansacParams& p) {
  RansacResult r;
  const std::size_t n = x1.size();
  r.inlier.assign(n, 0);
  if (n < 8 || x2.size() != n) return r;

  const double thr2 = p.threshold_px * p.threshold_px;
  eval::Rng rng(p.seed);
  std::vector<std::size_t> perm(n);
  Points2 s1(8);
  Points2 s2(8);
  Score best;
  Eigen::Matrix3d best_F = Eigen::Matrix3d::Zero();
  int limit = p.max_iters;
  int it = 0;
  for (; it < limit; ++it) {
    // 8 distinct indices: partial Fisher-Yates on a fresh identity permutation.
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    for (std::size_t k = 0; k < 8; ++k) {
      std::swap(perm[k], perm[k + rng.index(n - k)]);
      s1[k] = x1[perm[k]];
      s2[k] = x2[perm[k]];
    }
    Eigen::Matrix3d F;
    try {
      F = eight_point(s1, s2);
    } catch (const GeometryError&) {
      continue;  // degenerate sample (coincident points)
    }
    const Score s = evaluate(F, x1, x2, thr2, nullptr);
    if (s.better_than(best)) {
      best = s;
      best_F = F;
      limit =
          std::min(limit, needed_iterations(static_cast<double>(s.inliers) / static_cast<double>(n),
                                            p.confidence, p.max_iters));
    }
  }
  RansacResult r2 = refit_fundamental(x1, x2, best_F, p.threshold_px);
  r2.iterations = it;
  return r2;
}

RansacResult refit_fundamental(const Points2& x1, const Points2& x2, const Eigen::Matrix3d& F0,
                               double threshold_px) {
  RansacResult r;
  const std::size_t n = x1.size();
  r.inlier.assign(n, 0);
  r.F = F0;
  if (n < 8 || x2.size() != n || !(F0.norm() > 0.0)) return r;
  const double thr2 = threshold_px * threshold_px;
  std::vector<std::uint8_t> mask(n, 0);
  Score best = evaluate(F0, x1, x2, thr2, &mask);
  Eigen::Matrix3d best_F = F0;
  if (best.inliers < 8) {
    r.inliers = best.inliers;
    r.inlier = mask;
    return r;
  }

  // Refit on the inliers while that grows the inlier set.
  for (int round = 0; round < 3; ++round) {
    Points2 i1;
    Points2 i2;
    for (std::size_t k = 0; k < n; ++k) {
      if (mask[k] != 0) {
        i1.push_back(x1[k]);
        i2.push_back(x2[k]);
      }
    }
    Eigen::Matrix3d F;
    try {
      F = eight_point(i1, i2);
    } catch (const GeometryError&) {
      break;
    }
    std::vector<std::uint8_t> next(n, 0);
    const Score s = evaluate(F, x1, x2, thr2, &next);
    if (s.inliers < best.inliers) break;
    const bool grew = s.inliers > best.inliers;
    best = s;
    best_F = F;
    mask.swap(next);
    if (!grew) break;
  }
  r.F = best_F;
  r.inlier = mask;
  r.inliers = best.inliers;
  return r;
}

}  // namespace sfm::matching
