#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "sfm/matching/bf_matcher.hpp"

using sfm::matching::hamming256;
using sfm::matching::knn2;
using sfm::matching::match_ratio;

namespace {

// Descriptor with the lowest `bits` bits set (bits <= 256).
std::vector<std::uint32_t> desc_bits(int bits) {
  std::vector<std::uint32_t> d(8, 0);
  for (int b = 0; b < bits; ++b) d[static_cast<std::size_t>(b / 32)] |= 1U << (b % 32);
  return d;
}

std::vector<std::uint32_t> concat(const std::vector<std::vector<std::uint32_t>>& ds) {
  std::vector<std::uint32_t> out;
  for (const auto& d : ds) out.insert(out.end(), d.begin(), d.end());
  return out;
}

}  // namespace

TEST(Hamming, CountsDifferingBits) {
  EXPECT_EQ(hamming256(desc_bits(0).data(), desc_bits(0).data()), 0);
  EXPECT_EQ(hamming256(desc_bits(0).data(), desc_bits(256).data()), 256);
  EXPECT_EQ(hamming256(desc_bits(40).data(), desc_bits(100).data()), 60);
  std::vector<std::uint32_t> a(8, 0xAAAAAAAAU);
  std::vector<std::uint32_t> b(8, 0x55555555U);
  EXPECT_EQ(hamming256(a.data(), b.data()), 256);
  b[7] = 0xAAAAAAAAU;
  EXPECT_EQ(hamming256(a.data(), b.data()), 224);
}

TEST(BfMatcher, TopTwoWithTiesToTheLowerIndex) {
  const auto q = concat({desc_bits(10)});
  // Distances to the query: 5, 3, 3, 7 -> best index 1, second index 2 (tie: lower index first).
  const auto t = concat({desc_bits(15), desc_bits(13), desc_bits(7), desc_bits(17)});
  const auto k = knn2(q.data(), 1, t.data(), 4);
  ASSERT_EQ(k.size(), 1u);
  EXPECT_EQ(k[0].best, 1);
  EXPECT_EQ(k[0].best_distance, 3);
  EXPECT_EQ(k[0].second, 2);
  EXPECT_EQ(k[0].second_distance, 3);
  const auto one = knn2(q.data(), 1, t.data(), 1);
  EXPECT_EQ(one[0].best, 0);
  EXPECT_EQ(one[0].second, -1);
  EXPECT_TRUE(knn2(q.data(), 1, t.data(), 0)[0].best == -1);
}

TEST(BfMatcher, RatioAndMutual) {
  // q0 -> t0 (d 2) vs t1 (d 20): passes 0.75. q1 -> t2 (d 18) vs t1 (d 22): fails 0.75.
  // q2 -> t0 (d 6) vs t1 (d 14): passes ratio, but t0's nearest query is q0, so mutual drops it.
  const auto q = concat({desc_bits(100), desc_bits(142), desc_bits(106)});
  const auto t = concat({desc_bits(102), desc_bits(120), desc_bits(160), desc_bits(250)});
  const auto r = match_ratio(q.data(), 3, t.data(), 4, 0.75, false);
  ASSERT_EQ(r.size(), 2u);
  EXPECT_EQ(r[0].query, 0);
  EXPECT_EQ(r[0].train, 0);
  EXPECT_EQ(r[0].distance, 2);
  EXPECT_EQ(r[1].query, 2);
  EXPECT_EQ(r[1].train, 0);
  const auto m = match_ratio(q.data(), 3, t.data(), 4, 0.75, true);
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].query, 0);
  // Ratio 1.0 keeps q1 (18 < 22).
  EXPECT_EQ(match_ratio(q.data(), 3, t.data(), 4, 1.0, false).size(), 3u);
}

namespace {

// SIFT descriptor from 128 byte values (packed as in features.hpp).
std::vector<std::uint32_t> sift_desc(const std::vector<int>& bytes) {
  std::vector<std::uint32_t> d(32, 0);
  for (std::size_t k = 0; k < bytes.size(); ++k) {
    d[k / 4] |= static_cast<std::uint32_t>(bytes[k]) << (8 * (k % 4));
  }
  return d;
}

std::vector<int> bytes_with(std::size_t k, int v) {
  std::vector<int> b(128, 0);
  b[k] = v;
  return b;
}

}  // namespace

TEST(L2Matcher, SquaredDistanceOfBytes) {
  using sfm::matching::l2sq_u8x128;
  EXPECT_EQ(l2sq_u8x128(sift_desc(bytes_with(0, 0)).data(), sift_desc(bytes_with(0, 0)).data()), 0);
  EXPECT_EQ(l2sq_u8x128(sift_desc(bytes_with(5, 3)).data(), sift_desc(bytes_with(5, 7)).data()),
            16);
  // Different bytes in one word, and the largest possible distance.
  EXPECT_EQ(l2sq_u8x128(sift_desc(bytes_with(6, 2)).data(), sift_desc(bytes_with(7, 2)).data()), 8);
  const auto full = sift_desc(std::vector<int>(128, 255));
  EXPECT_EQ(l2sq_u8x128(full.data(), sift_desc(bytes_with(0, 0)).data()), 128 * 255 * 255);
}

TEST(L2Matcher, MatchesBruteForceAndRatioUsesDistancesNotSquares) {
  using sfm::features::kSiftWords;
  // Query = byte 0 at 0. Train: byte 0 at 10 (d = 10), 12 (d = 12), 10 again (tie -> lower index).
  const auto q = sift_desc(bytes_with(0, 0));
  const auto t = concat(
      {sift_desc(bytes_with(0, 10)), sift_desc(bytes_with(0, 12)), sift_desc(bytes_with(0, 10))});
  const auto k = knn2(q.data(), 1, t.data(), 3, kSiftWords);
  EXPECT_EQ(k[0].best, 0);
  EXPECT_EQ(k[0].best_distance, 100);
  EXPECT_EQ(k[0].second, 2);
  EXPECT_EQ(k[0].second_distance, 100);
  // Train (10, 12): distance ratio 10 / 12 = 0.833. Ratio 0.85 passes, 0.8 fails (squares: 0.69).
  const auto t2 = concat({sift_desc(bytes_with(0, 10)), sift_desc(bytes_with(0, 12))});
  EXPECT_EQ(match_ratio(q.data(), 1, t2.data(), 2, 0.85, false, kSiftWords).size(), 1u);
  EXPECT_EQ(match_ratio(q.data(), 1, t2.data(), 2, 0.8, false, kSiftWords).size(), 0u);
  EXPECT_THROW(knn2(q.data(), 1, t.data(), 1, 7), std::invalid_argument);
}
