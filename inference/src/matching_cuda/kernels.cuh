#pragma once
// Device entry points of the GPU matcher (no Eigen here: nvcc warns on Eigen's constexpr host
// functions, and CI builds with -Werror). The public API (matching_cuda.hpp) is in .cpp files.

#include <cstdint>
#include <vector>

namespace sfm::matching::cuda {

struct KnnPair {
  int query_offset;  // into the descriptor array (descriptors, not words)
  int n_query;
  int train_offset;
  int n_train;
  int out_offset;  // into the result array
};

// Top-2 (best, best distance, second, second distance) per (pair, query); desc = `words` words
// each: 8 = ORB (Hamming), 32 = SIFT (squared L2). Result: 4 ints per query, in out_offset order
// (out_total queries).
std::vector<int> knn2(const std::vector<unsigned int>& desc, const std::vector<KnnPair>& pairs,
                      int out_total, int max_query, int words);

struct RansacProblem {
  int offset;  // into the point arrays (points, not coordinates)
  int n;
  unsigned long long seed;
  int hyp_offset;
};

// Best hypothesis per problem: F (9 doubles, row-major, |F| = 1), all zero when no hypothesis was
// valid. x1 / x2: 2 doubles per point.
std::vector<double> ransac_best(const std::vector<double>& x1, const std::vector<double>& x2,
                                const std::vector<RansacProblem>& problems, int iters, double thr2);

}  // namespace sfm::matching::cuda
