#pragma once
// Brute-force matching: Hamming for ORB's 256-bit descriptors (8 x u32 each, contiguous), squared
// L2 for SIFT's 128 x u8 descriptors (32 x u32 each; features.hpp has the byte packing). Both are
// exact integer distances, so CPU and GPU agree bit for bit.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/matching/matches.hpp"

namespace sfm::matching {

// Hamming distance of two 256-bit descriptors.
inline int hamming256(const std::uint32_t* a, const std::uint32_t* b) {
  int n = 0;
  for (std::size_t w = 0; w < 4; ++w) {
    std::uint64_t x = 0;
    std::uint64_t y = 0;
    std::memcpy(&x, a + 2 * w, sizeof x);
    std::memcpy(&y, b + 2 * w, sizeof y);
    n += __builtin_popcountll(x ^ y);
  }
  return n;
}

// Squared L2 distance of two SIFT descriptors (<= 128 * 255^2 < 2^24).
inline int l2sq_u8x128(const std::uint32_t* a, const std::uint32_t* b) {
  int s = 0;
  for (std::size_t w = 0; w < 32; ++w) {
    for (int k = 0; k < 32; k += 8) {
      const int d = static_cast<int>((a[w] >> k) & 0xFFU) - static_cast<int>((b[w] >> k) & 0xFFU);
      s += d * d;
    }
  }
  return s;
}

// The two nearest train descriptors of one query. Index -1 if there is none (distance 257 for
// Hamming, kNoL2 for L2). Distances: Hamming, or squared L2 for SIFT.
inline constexpr int kNoL2 = 1 << 30;
struct Knn2 {
  int best = -1;
  int best_distance = 257;
  int second = -1;
  int second_distance = 257;
};

// For every query: the two nearest train descriptors. Ties go to the lower train index.
// `words` = descriptor width (kDescriptorWords: Hamming, kSiftWords: squared L2).
std::vector<Knn2> knn2(const std::uint32_t* query, std::size_t n_query, const std::uint32_t* train,
                       std::size_t n_train, int words = features::kDescriptorWords);

// Ratio test on distances: best < ratio * second (strict); for squared L2 the same test on the
// distances themselves, best_sq < ratio^2 * second_sq. A query with a single candidate passes.
inline bool passes_ratio(const Knn2& k, double ratio, int words) {
  if (k.second < 0) return true;
  if (words == features::kSiftWords) {
    return static_cast<double>(k.best_distance) <
           (ratio * ratio) * static_cast<double>(k.second_distance);
  }
  return k.best_distance < ratio * k.second_distance;
}

// Ratio test and, if `mutual`, the reverse check (the query is also the train descriptor's
// nearest query). Output in query order.
std::vector<Match> match_ratio(const std::uint32_t* query, std::size_t n_query,
                               const std::uint32_t* train, std::size_t n_train, double ratio,
                               bool mutual, int words = features::kDescriptorWords);

}  // namespace sfm::matching
