#pragma once
// Pair verification and tracks: putative matches of every image pair, F-RANSAC, union-find tracks.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/matching/fundamental.hpp"
#include "sfm/matching/matches.hpp"

namespace sfm::matching {

struct VerifyParams {
  double ratio = 0.75;
  bool mutual = false;
  double threshold_px = 3.0;
  double confidence = 0.999;
  int max_iters = 2000;
  int min_inliers = 30;
  std::uint64_t seed = 0;
  // "fundamental": 8-point F-RANSAC in pixels. "essential": 5-point E-RANSAC on K^-1-normalised
  // points (requires K, one per image); the stored F is K_j^-T E K_i^-1.
  std::string model = "fundamental";
  const std::vector<Eigen::Matrix3d>* K = nullptr;
};

// Seed of pair (i, j): a hash of (seed, i, j), so a pair's RANSAC does not depend on which other
// pairs were processed (or in which order, once pairs run in parallel).
std::uint64_t pair_seed(std::uint64_t seed, int i, int j);

// Robust two-view geometry of the putative correspondences of pair (i, j) by p.model (seeded with
// pair_seed); F is returned in pixels for both models.
RansacResult verify_geometry(const Points2& x1, const Points2& x2, int i, int j,
                             const VerifyParams& p);

// One pair: ratio-test matches i -> j, then verify_geometry. Returns true and fills `out` if at
// least min_inliers inliers remain (and the ratio test left at least that many matches).
bool verify_pair(const features::Features& fi, const features::Features& fj, int i, int j,
                 const VerifyParams& p, PairMatches& out);

// All pairs i < j in order; only verified pairs are returned.
// Image pairs to match, i < j, sorted. "exhaustive": all pairs. "sequential": (i, i+1 .. i+window)
// plus all pairs among the keyframes 0, s, 2s, ... (s = keyframe_stride; 0 = none). Throws
// std::invalid_argument for an unknown mode.
std::vector<std::pair<int, int>> select_pairs(int images, const std::string& mode, int window,
                                              int keyframe_stride);

// The given pairs (i < j) in order; only verified pairs are returned.
std::vector<PairMatches> verify_pairs(const std::vector<features::Features>& features,
                                      const std::vector<std::pair<int, int>>& pairs,
                                      const VerifyParams& p);

// verify_pairs over all pairs.
std::vector<PairMatches> verify_all_pairs(const std::vector<features::Features>& features,
                                          const VerifyParams& p);

// Union-find over (image, keypoint) nodes joined by the verified matches, strongest pairs first
// (most inliers). A merge that would put two keypoints of one image into one track is refused
// (the conflicting match is skipped), so a wrong match costs one link, not the whole track.
// Track ids are given in the order of each track's smallest (image, keypoint) node, so the result
// depends only on the inputs. keypoint_counts[i] = number of keypoints of image i.
Tracks build_tracks(const std::vector<PairMatches>& pairs,
                    const std::vector<std::size_t>& keypoint_counts);

}  // namespace sfm::matching
