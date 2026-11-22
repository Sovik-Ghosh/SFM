#pragma once
// Matching on the GPU (P4): brute-force Hamming top-2 for many image pairs in one batch (exactly
// equal to knn2), and F-RANSAC for many pairs in one batch (all max_iters hypotheses evaluated,
// deterministic for a seed; not equal to the CPU RANSAC, which samples differently and stops
// adaptively). verify_all_pairs_cuda = the bf_cpu stage with both parts on the GPU.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/matching/bf_matcher.hpp"
#include "sfm/matching/fundamental.hpp"
#include "sfm/matching/verify.hpp"

namespace sfm::matching {

// Top-2 of every descriptor of image pairs[k].first among those of pairs[k].second. out[k][q] ==
// knn2(...)[q] for the same inputs.
std::vector<std::vector<Knn2>> knn2_cuda(const std::vector<features::Features>& features,
                                         const std::vector<std::pair<int, int>>& pairs);

// One RANSAC per problem k (x1[k], x2[k], seed seeds[k]); threshold / max_iters from p (p.seed and
// p.confidence are not used: every one of the max_iters hypotheses is scored). Each problem's best
// hypothesis (most inliers, then least error, then lowest iteration) goes through
// refit_fundamental on the host. iterations = max_iters for problems with >= 8 points.
std::vector<RansacResult> ransac_fundamental_cuda(const std::vector<Points2>& x1,
                                                  const std::vector<Points2>& x2,
                                                  const std::vector<std::uint64_t>& seeds,
                                                  const RansacParams& p);

// The full stage on the GPU, same contract and pair seeds as verify_all_pairs.
// The top-2 runs in chunks of pairs holding at most max_queries_per_chunk query keypoints (16
// bytes each and direction on the host and the device); the result does not depend on it.
std::vector<PairMatches> verify_pairs_cuda(const std::vector<features::Features>& features,
                                           const std::vector<std::pair<int, int>>& pairs,
                                           const VerifyParams& p,
                                           std::size_t max_queries_per_chunk = std::size_t{32}
                                                                               << 20);
std::vector<PairMatches> verify_all_pairs_cuda(const std::vector<features::Features>& features,
                                               const VerifyParams& p);

}  // namespace sfm::matching
