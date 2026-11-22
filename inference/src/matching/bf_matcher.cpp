#include "sfm/matching/bf_matcher.hpp"

#include <stdexcept>
#include <string>

namespace sfm::matching {

namespace {

template <class Distance>
std::vector<Knn2> knn2_impl(const std::uint32_t* query, std::size_t n_query,
                            const std::uint32_t* train, std::size_t n_train, std::size_t words,
                            int none, Distance dist) {
  std::vector<Knn2> out(n_query);
  for (std::size_t q = 0; q < n_query; ++q) {
    Knn2 k;
    k.best_distance = none;
    k.second_distance = none;
    const std::uint32_t* dq = query + words * q;
    for (std::size_t t = 0; t < n_train; ++t) {
      const int d = dist(dq, train + words * t);
      // Strict comparisons: an equal distance never replaces an earlier (lower) index.
      if (d < k.best_distance) {
        k.second = k.best;
        k.second_distance = k.best_distance;
        k.best = static_cast<int>(t);
        k.best_distance = d;
      } else if (d < k.second_distance) {
        k.second = static_cast<int>(t);
        k.second_distance = d;
      }
    }
    out[q] = k;
  }
  return out;
}

}  // namespace

std::vector<Knn2> knn2(const std::uint32_t* query, std::size_t n_query, const std::uint32_t* train,
                       std::size_t n_train, int words) {
  if (words == features::kDescriptorWords) {
    return knn2_impl(query, n_query, train, n_train, 8, 257, hamming256);
  }
  if (words == features::kSiftWords) {
    return knn2_impl(query, n_query, train, n_train, 32, kNoL2, l2sq_u8x128);
  }
  throw std::invalid_argument("knn2: unsupported descriptor width " + std::to_string(words));
}

std::vector<Match> match_ratio(const std::uint32_t* query, std::size_t n_query,
                               const std::uint32_t* train, std::size_t n_train, double ratio,
                               bool mutual, int words) {
  const auto fwd = knn2(query, n_query, train, n_train, words);
  std::vector<Knn2> back;
  if (mutual) back = knn2(train, n_train, query, n_query, words);
  std::vector<Match> out;
  for (std::size_t q = 0; q < fwd.size(); ++q) {
    const Knn2& k = fwd[q];
    if (k.best < 0) continue;
    if (!passes_ratio(k, ratio, words)) continue;
    if (mutual && back[static_cast<std::size_t>(k.best)].best != static_cast<int>(q)) continue;
    out.push_back({static_cast<int>(q), k.best, k.best_distance});
  }
  return out;
}

}  // namespace sfm::matching
