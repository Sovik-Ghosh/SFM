#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include "sfm/core/parallel.hpp"
#include "sfm/matching/bf_matcher.hpp"
#include "sfm/matching/five_point.hpp"
#include "sfm/matching/verify.hpp"

namespace sfm::matching {

std::uint64_t pair_seed(std::uint64_t seed, int i, int j) {
  // splitmix64 finaliser over the three values.
  std::uint64_t z = seed ^ (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(j);
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

RansacResult verify_geometry(const Points2& x1, const Points2& x2, int i, int j,
                             const VerifyParams& p) {
  const std::uint64_t seed = pair_seed(p.seed, i, j);
  if (p.model != "essential") {
    return ransac_fundamental(x1, x2, {p.threshold_px, p.confidence, p.max_iters, seed});
  }
  if (p.K == nullptr || static_cast<std::size_t>(std::max(i, j)) >= p.K->size()) {
    throw std::invalid_argument("verify: model \"essential\" needs K for every image");
  }
  const Eigen::Matrix3d& Ki = (*p.K)[static_cast<std::size_t>(i)];
  const Eigen::Matrix3d& Kj = (*p.K)[static_cast<std::size_t>(j)];
  const Eigen::Matrix3d Ki_inv = Ki.inverse();
  const Eigen::Matrix3d Kj_inv = Kj.inverse();
  Points2 n1;
  Points2 n2;
  n1.reserve(x1.size());
  n2.reserve(x2.size());
  for (std::size_t k = 0; k < x1.size(); ++k) {
    n1.push_back((Ki_inv * x1[k].homogeneous()).hnormalized());
    n2.push_back((Kj_inv * x2[k].homogeneous()).hnormalized());
  }
  EssentialRansacParams ep;
  ep.threshold_px = p.threshold_px;
  ep.focal = 0.5 * (Ki(0, 0) + Kj(0, 0));
  ep.confidence = p.confidence;
  ep.max_iters = p.max_iters;
  ep.seed = seed;
  const EssentialRansacResult e = ransac_essential(n1, n2, ep);
  RansacResult r;
  r.inlier = e.inlier;
  r.inliers = e.inliers;
  r.iterations = e.iterations;
  if (e.E.norm() > 0.0) {
    r.F = Kj_inv.transpose() * e.E * Ki_inv;
    r.F /= r.F.norm();
  }
  return r;
}

bool verify_pair(const features::Features& fi, const features::Features& fj, int i, int j,
                 const VerifyParams& p, PairMatches& out) {
  if (fi.words != fj.words) throw std::invalid_argument("verify_pair: descriptor widths differ");
  const auto m = match_ratio(fi.desc.data(), fi.size(), fj.desc.data(), fj.size(), p.ratio,
                             p.mutual, fi.words);
  if (m.size() < static_cast<std::size_t>(p.min_inliers) || m.size() < 8) return false;
  Points2 x1;
  Points2 x2;
  x1.reserve(m.size());
  x2.reserve(m.size());
  for (const Match& k : m) {
    const auto q = static_cast<std::size_t>(k.query);
    const auto t = static_cast<std::size_t>(k.train);
    x1.emplace_back(fi.xy[2 * q], fi.xy[2 * q + 1]);
    x2.emplace_back(fj.xy[2 * t], fj.xy[2 * t + 1]);
  }
  const RansacResult r = verify_geometry(x1, x2, i, j, p);
  if (r.inliers < p.min_inliers) return false;
  out.i = i;
  out.j = j;
  out.F = r.F;
  out.idx.clear();
  out.idx.reserve(2 * static_cast<std::size_t>(r.inliers));
  for (std::size_t k = 0; k < m.size(); ++k) {
    if (r.inlier[k] != 0) {
      out.idx.push_back(m[k].query);
      out.idx.push_back(m[k].train);
    }
  }
  return true;
}

std::vector<std::pair<int, int>> select_pairs(int images, const std::string& mode, int window,
                                              int keyframe_stride) {
  std::vector<std::pair<int, int>> out;
  if (mode == "exhaustive") {
    for (int i = 0; i < images; ++i) {
      for (int j = i + 1; j < images; ++j) out.emplace_back(i, j);
    }
    return out;
  }
  if (mode != "sequential") throw std::invalid_argument("select_pairs: unknown mode " + mode);
  for (int i = 0; i < images; ++i) {
    for (int j = i + 1; j <= std::min(images - 1, i + window); ++j) out.emplace_back(i, j);
  }
  if (keyframe_stride > 0) {
    for (int i = 0; i < images; i += keyframe_stride) {
      for (int j = i + keyframe_stride; j < images; j += keyframe_stride) {
        if (j - i > window) out.emplace_back(i, j);  // closer pairs are already in the window
      }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<PairMatches> verify_pairs(const std::vector<features::Features>& features,
                                      const std::vector<std::pair<int, int>>& pairs,
                                      const VerifyParams& p) {
  // Pairs are independent and seeded per pair: verified on all cores, kept in pair order.
  std::vector<PairMatches> found(pairs.size());
  std::vector<char> ok(pairs.size(), 0);
  core::parallel_for(pairs.size(), [&](std::size_t k) {
    const auto [i, j] = pairs[k];
    ok[k] = verify_pair(features[static_cast<std::size_t>(i)],
                        features[static_cast<std::size_t>(j)], i, j, p, found[k])
                ? 1
                : 0;
  });
  std::vector<PairMatches> out;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    if (ok[k] != 0) out.push_back(std::move(found[k]));
  }
  return out;
}

std::vector<PairMatches> verify_all_pairs(const std::vector<features::Features>& features,
                                          const VerifyParams& p) {
  return verify_pairs(features, select_pairs(static_cast<int>(features.size()), "exhaustive", 0, 0),
                      p);
}

}  // namespace sfm::matching
