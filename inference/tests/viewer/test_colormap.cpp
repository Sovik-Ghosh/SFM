#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "sfm/viewer/colormap.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(Colormap, ViridisEndpointsClampAndOrder) {
  EXPECT_EQ(viridis(0.0F), (Rgb{68, 1, 84}));
  EXPECT_EQ(viridis(1.0F), (Rgb{253, 231, 37}));
  EXPECT_EQ(viridis(0.5F), (Rgb{33, 145, 140}));
  EXPECT_EQ(viridis(-3.0F), viridis(0.0F));
  EXPECT_EQ(viridis(7.0F), viridis(1.0F));
  EXPECT_EQ(viridis(std::numeric_limits<float>::quiet_NaN()), viridis(0.0F));
  // Luminance grows monotonically (the property that makes it readable).
  float last = -1.0F;
  for (int k = 0; k <= 100; ++k) {
    const Rgb c = viridis(static_cast<float>(k) / 100.0F);
    const float lum = 0.2126F * static_cast<float>(c[0]) + 0.7152F * static_cast<float>(c[1]) +
                      0.0722F * static_cast<float>(c[2]);
    EXPECT_GE(lum, last - 0.5F) << k;
    last = lum;
  }
}

TEST(Colormap, RobustRangeIgnoresOutliers) {
  std::vector<float> v;
  v.reserve(101);
  for (int k = 0; k < 100; ++k) v.push_back(static_cast<float>(k));
  v.push_back(1e6F);  // one outlier
  const auto r = robust_range(v);
  EXPECT_NEAR(r[0], 5.0F, 1.0F);
  EXPECT_NEAR(r[1], 95.0F, 1.0F);
  EXPECT_EQ(robust_range({}), (std::array<float, 2>{0.0F, 1.0F}));
  EXPECT_EQ(robust_range({3.0F, 3.0F}), (std::array<float, 2>{3.0F, 4.0F}));
  EXPECT_EQ(normalise(2.0F, 0.0F, 4.0F), 0.5F);
  EXPECT_EQ(normalise(-1.0F, 0.0F, 4.0F), 0.0F);
  EXPECT_EQ(normalise(9.0F, 0.0F, 4.0F), 1.0F);
}
