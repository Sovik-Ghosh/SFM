#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <set>

#include "sfm/data/image_io.hpp"
#include "sfm/features/exact.hpp"
#include "sfm/features/orb.hpp"

#include "image_util.hpp"

namespace fs = std::filesystem;
using sfm::features::detect_orb;
using sfm::features::ic_angle;
using sfm::features::level_quotas;
using sfm::features::OrbParams;
using sfm::features::rbrief;
using sfm::features::rbrief_pattern;
using sfm::features::test::make_image;
using sfm::features::test::rotate90;
using sfm::features::test::texture;

namespace {

double angle_diff(double a, double b) {
  const double d = std::fmod(std::fabs(a - b), 360.0);
  return std::min(d, 360.0 - d);
}

int hamming(const std::uint32_t* a, const std::uint32_t* b) {
  int n = 0;
  for (int w = 0; w < 8; ++w) n += static_cast<int>(std::bitset<32>(a[w] ^ b[w]).count());
  return n;
}

}  // namespace

TEST(Orb, LevelQuotasSumToMaxAndShrink) {
  OrbParams p;
  const auto q = level_quotas(p);
  ASSERT_EQ(q.size(), 8u);
  EXPECT_EQ(std::accumulate(q.begin(), q.end(), 0), 2000);
  for (std::size_t l = 1; l + 1 < q.size(); ++l) EXPECT_LT(q[l], q[l - 1]);
  EXPECT_EQ(q[0], 434);  // 2000 * (1 - 1/1.2) / (1 - 1.2^-8) = 434.4
}

TEST(Orb, PatternIsInRangeAndFixed) {
  const auto& p = rbrief_pattern();
  std::set<std::array<std::int8_t, 4>> distinct(p.begin(), p.end());
  EXPECT_GT(distinct.size(), 250u);
  for (const auto& t : p) {
    for (std::int8_t v : t) EXPECT_LE(std::abs(v), 13);
    EXPECT_FALSE(t[0] == t[2] && t[1] == t[3]);
  }
  EXPECT_EQ(&rbrief_pattern(), &p);
}

TEST(Orb, IntensityCentroidPointsToTheBrightSide) {
  const auto right = make_image(41, 41, [](int x, int) { return x > 20 ? 200 : 20; });
  EXPECT_NEAR(ic_angle(right, 20, 20), 0.0, 1e-3);
  const auto down = make_image(41, 41, [](int, int y) { return y > 20 ? 200 : 20; });
  EXPECT_NEAR(ic_angle(down, 20, 20), 90.0, 1e-3);  // image y points down
  const auto left = make_image(41, 41, [](int x, int) { return x < 20 ? 200 : 20; });
  EXPECT_NEAR(ic_angle(left, 20, 20), 180.0, 1e-3);
}

TEST(Orb, RotatingThePatchRotatesTheAngleAndKeepsTheDescriptor) {
  const int n = 101;
  const auto img = make_image(n, n, texture);
  const auto rot = rotate90(img);
  const int c = 50;  // the centre maps onto itself
  const float a0 = ic_angle(img, c, c);
  const float a1 = ic_angle(rot, c, c);
  EXPECT_LT(angle_diff(a1, a0 + 90.0), 1.0) << a0 << " " << a1;

  const auto steer = [](float deg, double& co, double& si) {
    co = std::cos(deg * M_PI / 180.0);
    si = std::sin(deg * M_PI / 180.0);
  };
  double c0 = 0;
  double s0 = 0;
  double c1 = 0;
  double s1 = 0;
  steer(a0, c0, s0);
  steer(a1, c1, s1);
  std::uint32_t d0[8];
  std::uint32_t d1[8];
  rbrief(img, c, c, c0, s0, d0);
  rbrief(rot, c, c, c1, s1, d1);
  EXPECT_LE(hamming(d0, d1), 20);
  // Without steering the rotated patch is a different patch.
  rbrief(rot, c, c, c0, s0, d1);
  EXPECT_GT(hamming(d0, d1), 60);
  // Steering by the moments is steering by the angle (up to rounding of rotated offsets).
  std::int64_t m10 = 0;
  std::int64_t m01 = 0;
  sfm::features::exact::moments(img.pixels.data(), img.width, c, c, m10, m01);
  EXPECT_NEAR(sfm::features::angle_deg(m10, m01), a0, 1e-4);
  std::uint32_t dm[8];
  rbrief(img, c, c, m10, m01, dm);
  EXPECT_LE(hamming(d0, dm), 2);
}

TEST(Orb, BunnyImageDetectsWithinMaskDeterministically) {
  const fs::path root = fs::path(SFM_SOURCE_DIR) / "data" / "bunny";
  const auto gray = sfm::data::load_gray(root / "images" / "0012.ppm");
  const auto mask = sfm::data::load_mask(root / "silhouettes" / "0012.pgm");
  OrbParams p;
  const auto f = detect_orb(gray, &mask, p);
  EXPECT_GT(f.size(), 1500u);
  EXPECT_LE(f.size(), 2000u);
  ASSERT_EQ(f.xy.size(), 2 * f.size());
  ASSERT_EQ(f.desc.size(), 8 * f.size());
  ASSERT_EQ(f.octave.size(), f.size());
  std::set<int> octaves;
  for (std::size_t i = 0; i < f.size(); ++i) {
    const int x = static_cast<int>(std::lround(f.xy[2 * i]));
    const int y = static_cast<int>(std::lround(f.xy[2 * i + 1]));
    ASSERT_GE(x, 0);
    ASSERT_LT(x, gray.width);
    ASSERT_GE(y, 0);
    ASSERT_LT(y, gray.height);
    // Allowed on the level's (nearest-neighbour) mask; at level 0 the pixel itself is in the mask.
    if (f.octave[i] == 0) {
      EXPECT_NE(sfm::features::test::at(mask, x, y), 0);
    }
    EXPECT_GE(f.angle[i], 0.0F);
    EXPECT_LT(f.angle[i], 360.0F);
    octaves.insert(f.octave[i]);
  }
  EXPECT_EQ(octaves.size(), 8u);
  const auto again = detect_orb(gray, &mask, p);
  EXPECT_EQ(again.xy, f.xy);
  EXPECT_EQ(again.desc, f.desc);
}
