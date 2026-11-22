#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "sfm/features/sift.hpp"
#include "sfm/matching/bf_matcher.hpp"

#include "image_util.hpp"

using sfm::features::detect_sift;
using sfm::features::Features;
using sfm::features::SiftParams;
using sfm::features::test::make_image;
using sfm::features::test::rotate90;
using sfm::features::test::texture;

namespace {

int byte_of(const Features& f, std::size_t i, int k) {
  return static_cast<int>((f.descriptor(i)[k / 4] >> (8 * (k % 4))) & 0xFFU);
}

}  // namespace

TEST(Sift, GaussianTapsAreNormalisedAndSymmetric) {
  for (const double s : {0.6, 1.25, 3.7}) {
    const auto t = sfm::features::gaussian_taps(s);
    const int r = static_cast<int>(t.size() / 2);
    EXPECT_EQ(r, std::max(1, static_cast<int>(std::ceil(4.0 * s))));
    EXPECT_NEAR(std::accumulate(t.begin(), t.end(), 0.0), 1.0, 1e-6);
    for (int k = 0; k < r; ++k) EXPECT_EQ(t[static_cast<std::size_t>(k)], t[t.size() - 1 - k]);
  }
}

TEST(Sift, FindsTheCentreOfABlob) {
  const double cx = 47.3;
  const double cy = 50.6;
  const auto img = make_image(96, 96, [&](int x, int y) {
    const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
    return static_cast<int>(std::lround(40 + 180 * std::exp(-r2 / (2 * 5.0 * 5.0))));
  });
  const Features f = detect_sift(img, nullptr, SiftParams{});
  ASSERT_GT(f.size(), 0u);
  EXPECT_EQ(f.words, sfm::features::kSiftWords);
  // The strongest location is the blob itself.
  EXPECT_NEAR(f.xy[0], cx, 0.3);
  EXPECT_NEAR(f.xy[1], cy, 0.3);
  EXPECT_GT(f.response[0], SiftParams{}.peak_threshold);
}

TEST(Sift, DescriptorIsRootSiftQuantised) {
  const auto img = make_image(128, 128, texture);
  const Features f = detect_sift(img, nullptr, SiftParams{});
  ASSERT_GT(f.size(), 20u);
  for (std::size_t i = 0; i < f.size(); ++i) {
    // RootSIFT: sum v^2 = sum (L1-normalised) = 1, quantised as 512 v.
    double s2 = 0.0;
    for (int k = 0; k < 128; ++k) s2 += std::pow(byte_of(f, i, k) / 512.0, 2);
    EXPECT_NEAR(s2, 1.0, 0.03) << "keypoint " << i;
    EXPECT_GE(f.angle[i], 0.0F);
    EXPECT_LT(f.angle[i], 360.0F);
  }
}

TEST(Sift, RotatingTheImageBy90DegreesRotatesKeypointsAndKeepsDescriptors) {
  const auto img = make_image(128, 128, texture);
  const Features a = detect_sift(img, nullptr, SiftParams{});
  const Features b = detect_sift(rotate90(img), nullptr, SiftParams{});
  const int n = img.width;
  int central = 0;
  int found = 0;
  int angle_ok = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float u = a.xy[2 * i];
    const float v = a.xy[2 * i + 1];
    if (std::hypot(u - 63.5, v - 63.5) > 40.0) continue;
    ++central;
    const float ru = static_cast<float>(n - 1) - v;  // rotate90: (u, v) -> (n - 1 - v, u)
    const float rv = u;
    for (std::size_t j = 0; j < b.size(); ++j) {
      if (std::hypot(b.xy[2 * j] - ru, b.xy[2 * j + 1] - rv) > 0.3) continue;
      double d = std::fmod(b.angle[j] - a.angle[i] - 90.0 + 720.0, 360.0);
      d = std::min(d, 360.0 - d);
      if (d < 3.0) {
        ++found;
        angle_ok += sfm::matching::l2sq_u8x128(a.descriptor(i), b.descriptor(j)) < 30000 ? 1 : 0;
        break;
      }
    }
  }
  ASSERT_GT(central, 20);
  EXPECT_GE(found, 0.8 * central);
  EXPECT_GE(angle_ok, 0.9 * found);
  // And they match: the ratio test pairs most keypoints with their rotated selves.
  const auto m = sfm::matching::match_ratio(a.desc.data(), a.size(), b.desc.data(), b.size(), 0.8,
                                            true, sfm::features::kSiftWords);
  int correct = 0;
  for (const auto& mm : m) {
    const auto q = static_cast<std::size_t>(mm.query);
    const auto t = static_cast<std::size_t>(mm.train);
    correct += std::hypot(b.xy[2 * t] - (static_cast<float>(n - 1) - a.xy[2 * q + 1]),
                          b.xy[2 * t + 1] - a.xy[2 * q]) < 1.0
                   ? 1
                   : 0;
  }
  ASSERT_GT(m.size(), 20u);
  EXPECT_GE(correct, 0.9 * static_cast<double>(m.size()));
}

TEST(Sift, MaskAndFeatureLimit) {
  const auto img = make_image(128, 128, texture);
  const auto left = make_image(128, 128, [](int x, int /*y*/) { return x < 64 ? 255 : 0; });
  const Features m = detect_sift(img, &left, SiftParams{});
  ASSERT_GT(m.size(), 0u);
  for (std::size_t i = 0; i < m.size(); ++i) EXPECT_LT(std::floor(m.xy[2 * i] + 0.5F), 64.0F);
  const auto none = make_image(128, 128, [](int, int) { return 0; });
  EXPECT_EQ(detect_sift(img, &none, SiftParams{}).size(), 0u);

  // max_features keeps the strongest locations: the output is a prefix of the full one.
  const Features full = detect_sift(img, nullptr, SiftParams{});
  SiftParams p;
  p.max_features = 10;
  const Features top = detect_sift(img, nullptr, p);
  ASSERT_GE(top.size(), 10u);
  ASSERT_LE(top.size(), 20u);  // <= 2 orientations each
  EXPECT_EQ(std::vector<float>(full.xy.begin(),
                               full.xy.begin() + static_cast<std::ptrdiff_t>(top.xy.size())),
            top.xy);
  EXPECT_EQ(
      std::vector<std::uint32_t>(full.desc.begin(),
                                 full.desc.begin() + static_cast<std::ptrdiff_t>(top.desc.size())),
      top.desc);
  for (std::size_t i = 1; i < top.size(); ++i) EXPECT_LE(top.response[i], top.response[i - 1]);
}
