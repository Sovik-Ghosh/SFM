#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

#include "sfm/features/fast.hpp"

#include "image_util.hpp"

using sfm::features::Corner;
using sfm::features::detect_fast;
using sfm::features::fast_score;
using sfm::features::nms_3x3;
using sfm::features::test::make_image;

namespace {

// Reference definition: some arc of 9 contiguous circle pixels all > c + t or all < c - t.
bool is_corner_reference(const sfm::data::Image8& img, int x, int y, int t) {
  static constexpr std::array<std::array<int, 2>, 16> kC = {{{0, -3},
                                                             {1, -3},
                                                             {2, -2},
                                                             {3, -1},
                                                             {3, 0},
                                                             {3, 1},
                                                             {2, 2},
                                                             {1, 3},
                                                             {0, 3},
                                                             {-1, 3},
                                                             {-2, 2},
                                                             {-3, 1},
                                                             {-3, 0},
                                                             {-3, -1},
                                                             {-2, -2},
                                                             {-1, -3}}};
  const int c = sfm::features::test::at(img, x, y);
  for (std::size_t s = 0; s < 16; ++s) {
    bool bright = true;
    bool dark = true;
    for (std::size_t k = 0; k < 9; ++k) {
      const auto& o = kC[(s + k) % 16];
      const int p = sfm::features::test::at(img, x + o[0], y + o[1]);
      bright = bright && p > c + t;
      dark = dark && p < c - t;
    }
    if (bright || dark) return true;
  }
  return false;
}

}  // namespace

TEST(Fast, OneBrightDotIsOneCorner) {
  const auto img = make_image(13, 13, [](int x, int y) { return (x == 6 && y == 6) ? 200 : 50; });
  const auto c = detect_fast(img, nullptr, 20, 3);
  ASSERT_EQ(c.size(), 1u);
  EXPECT_EQ(c[0].x, 6);
  EXPECT_EQ(c[0].y, 6);
  EXPECT_EQ(c[0].score, 150);
}

TEST(Fast, FlatAndStraightEdgeGiveNone) {
  const auto flat = make_image(20, 20, [](int, int) { return 90; });
  EXPECT_TRUE(detect_fast(flat, nullptr, 5, 3).empty());
  const auto edge = make_image(20, 20, [](int x, int) { return x < 10 ? 20 : 220; });
  EXPECT_TRUE(detect_fast(edge, nullptr, 5, 3).empty());
}

TEST(Fast, SquareCornerIsFound) {
  // Bright square [10, 30)^2: its corners are FAST corners (seen from outside, 12 of 16 are dark).
  const auto img = make_image(
      40, 40, [](int x, int y) { return (x >= 10 && x < 30 && y >= 10 && y < 30) ? 200 : 30; });
  const auto c = detect_fast(img, nullptr, 20, 3);
  // Pixels next to a square corner score the same; NMS keeps the first in raster order, so each
  // corner gives exactly one keypoint within 2 px of it.
  ASSERT_EQ(c.size(), 4u);
  std::set<int> seen;
  for (const Corner& k : c) {
    const int cx = k.x < 20 ? 10 : 29;
    const int cy = k.y < 20 ? 10 : 29;
    EXPECT_LE(std::abs(k.x - cx), 2) << k.x;
    EXPECT_LE(std::abs(k.y - cy), 2) << k.y;
    seen.insert(cx * 100 + cy);
  }
  EXPECT_EQ(seen.size(), 4u);
}

TEST(Fast, ScoreMatchesTheDefinitionForEveryThreshold) {
  const auto img = make_image(32, 32, [](int x, int y) {
    return static_cast<int>((x * 7919U + y * 104729U + x * y * 31U) % 256U);
  });
  for (int y = 3; y < 29; ++y) {
    for (int x = 3; x < 29; ++x) {
      const int s = fast_score(img, x, y);
      for (int t : {0, 5, 20, 60, 120}) {
        ASSERT_EQ(s > t, is_corner_reference(img, x, y, t)) << x << "," << y << " t=" << t;
      }
    }
  }
}

TEST(Fast, MaskAndBorderAreRespected) {
  const auto img = make_image(
      40, 40, [](int x, int y) { return (x >= 10 && x < 30 && y >= 10 && y < 30) ? 200 : 30; });
  const auto mask = make_image(40, 40, [](int x, int) { return x < 20 ? 255 : 0; });
  const auto c = detect_fast(img, &mask, 20, 3);
  ASSERT_EQ(c.size(), 2u);
  for (const Corner& k : c) EXPECT_EQ(k.x, 10);
  for (const Corner& k : detect_fast(img, nullptr, 20, 12)) {
    EXPECT_TRUE(k.x >= 12 && k.x < 28 && k.y >= 12 && k.y < 28) << k.x << "," << k.y;
  }
  EXPECT_EQ(detect_fast(img, nullptr, 20, 10).size(), 4u);
  EXPECT_THROW(detect_fast(img, nullptr, 20, 2), std::invalid_argument);
}

TEST(Nms, KeepsLocalMaximaAndOneOfAPlateau) {
  // 5x4 map: a peak of 9 next to 5; a plateau of three 4s; an isolated 1.
  const std::vector<int> s = {0, 0, 0, 0, 0,  //
                              0, 5, 9, 0, 0,  //
                              0, 0, 0, 0, 1,  //
                              4, 4, 4, 0, 0};
  const auto c = nms_3x3(s, 5, 4);
  ASSERT_EQ(c.size(), 3u);
  EXPECT_EQ((std::array<int, 3>{c[0].x, c[0].y, c[0].score}), (std::array<int, 3>{2, 1, 9}));
  EXPECT_EQ((std::array<int, 3>{c[1].x, c[1].y, c[1].score}), (std::array<int, 3>{4, 2, 1}));
  EXPECT_EQ((std::array<int, 3>{c[2].x, c[2].y, c[2].score}), (std::array<int, 3>{0, 3, 4}));
  EXPECT_THROW(nms_3x3(s, 4, 4), std::invalid_argument);
}

TEST(Harris, CornerBeatsEdgeBeatsFlat) {
  const auto corner =
      make_image(21, 21, [](int x, int y) { return (x >= 10 && y >= 10) ? 200 : 30; });
  const auto edge = make_image(21, 21, [](int x, int) { return x >= 10 ? 200 : 30; });
  const auto flat = make_image(21, 21, [](int, int) { return 77; });
  const float rc = sfm::features::harris_response(corner, 10, 10);
  const float re = sfm::features::harris_response(edge, 10, 10);
  const float rf = sfm::features::harris_response(flat, 10, 10);
  EXPECT_GT(rc, 0.0F);
  EXPECT_LT(re, 0.0F);  // one large eigenvalue: det ~ 0, trace term negative
  EXPECT_EQ(rf, 0.0F);
}
