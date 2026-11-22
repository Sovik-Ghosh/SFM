#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sfm/matching/verify.hpp"

using sfm::matching::build_tracks;
using sfm::matching::PairMatches;

namespace {

PairMatches pair(int i, int j, std::vector<std::int32_t> idx) {
  PairMatches p;
  p.i = i;
  p.j = j;
  p.idx = std::move(idx);
  return p;
}

}  // namespace

TEST(Tracks, ChainsMergeAndIdsFollowTheSmallestNode) {
  // (0,5)-(1,2), (1,2)-(2,7): one track of three. (1,0)-(2,1): a second track.
  const auto t = build_tracks({pair(1, 2, {2, 7, 0, 1}), pair(0, 1, {5, 2})}, {10, 10, 10});
  EXPECT_EQ(t.count, 2);
  const std::vector<std::int32_t> want = {0, 0, 5, 0, 1, 2, 0, 2, 7,  //
                                          1, 1, 0, 1, 2, 1};
  EXPECT_EQ(t.obs, want);
  EXPECT_EQ(t.size(), 5u);
}

TEST(Tracks, AConflictingMatchIsRefusedNotTheTrack) {
  // (0,1)-(1,1) and (0,3)-(1,4) from the strongest pair; then (0,2)-(2,1) (pair (0,2) before (1,2)
  // on a tie); (1,1)-(2,1) would put image 0 twice into one track -> refused. Three tracks stay.
  const auto t =
      build_tracks({pair(0, 1, {1, 1, 3, 4}), pair(1, 2, {1, 1}), pair(0, 2, {2, 1})}, {5, 5, 5});
  EXPECT_EQ(t.count, 3);
  const std::vector<std::int32_t> want = {0, 0, 1, 0, 1, 1,   // (0,1)-(1,1)
                                          1, 0, 2, 1, 2, 1,   // (0,2)-(2,1)
                                          2, 0, 3, 2, 1, 4};  // (0,3)-(1,4)
  EXPECT_EQ(t.obs, want);
}

TEST(Tracks, OrderOfPairsDoesNotMatterAndBadIndicesThrow) {
  const std::vector<PairMatches> a = {pair(0, 1, {0, 0, 1, 1}), pair(1, 2, {0, 3, 1, 2}),
                                      pair(0, 2, {4, 4})};
  const std::vector<PairMatches> b = {a[2], a[1], a[0]};
  EXPECT_EQ(build_tracks(a, {5, 5, 5}).obs, build_tracks(b, {5, 5, 5}).obs);
  EXPECT_THROW(build_tracks({pair(0, 3, {0, 0})}, {5, 5, 5}), std::out_of_range);
  EXPECT_THROW(build_tracks({pair(0, 1, {0, 5})}, {5, 5, 5}), std::out_of_range);
}

TEST(SelectPairs, ExhaustiveAndSequential) {
  using sfm::matching::select_pairs;
  EXPECT_EQ(select_pairs(4, "exhaustive", 0, 0).size(), 6u);
  // window 2: (0,1) (0,2) (1,2) (1,3) (2,3) (2,4) (3,4) (3,5) (4,5); keyframes 0, 3: (0,3) not in
  // window -> added.
  const auto s = select_pairs(6, "sequential", 2, 3);
  const std::vector<std::pair<int, int>> want = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3},
                                                 {2, 3}, {2, 4}, {3, 4}, {3, 5}, {4, 5}};
  EXPECT_EQ(s, want);
  EXPECT_EQ(select_pairs(6, "sequential", 2, 0).size(), 9u);  // no keyframes
  // Linear growth: 1000 images, window 10, stride 10 -> 1000 * 10 - 55 + (100 * 99 / 2 - 99) pairs.
  EXPECT_EQ(select_pairs(1000, "sequential", 10, 10).size(), 1000u * 10u - 55u + 4950u - 99u);
  EXPECT_THROW(select_pairs(3, "random", 1, 1), std::invalid_argument);
}
