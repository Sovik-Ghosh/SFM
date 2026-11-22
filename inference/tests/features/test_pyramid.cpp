#include <gtest/gtest.h>

#include <set>
#include <utility>
#include <vector>

#include "sfm/features/pyramid.hpp"

#include "image_util.hpp"

using sfm::features::build_pyramid;
using sfm::features::gaussian_blur7;
using sfm::features::Resample;
using sfm::features::resize;
using sfm::features::test::at;
using sfm::features::test::make_image;

TEST(Pyramid, BunnyLevelSizes) {
  const auto base = make_image(1024, 768, [](int x, int y) { return (x + y) % 256; });
  const auto levels = build_pyramid(base, 8, 1.2);
  const std::vector<std::pair<int, int>> want = {{1024, 768}, {853, 640}, {711, 533}, {593, 444},
                                                 {494, 370},  {412, 309}, {343, 257}, {286, 214}};
  ASSERT_EQ(levels.size(), want.size());
  for (std::size_t l = 0; l < want.size(); ++l) {
    EXPECT_EQ(levels[l].width, want[l].first) << l;
    EXPECT_EQ(levels[l].height, want[l].second) << l;
  }
  EXPECT_EQ(levels[0].pixels, base.pixels);
  EXPECT_THROW(build_pyramid(base, 0, 1.2), std::invalid_argument);
  EXPECT_THROW(build_pyramid(base, 3, 1.0), std::invalid_argument);
}

TEST(Pyramid, BilinearHalvingAveragesPairs) {
  // Columns 0,1 -> mean; constant rows. Halving samples exactly between two source pixels.
  const auto img = make_image(8, 2, [](int x, int) { return x * 10; });
  const auto half = resize(img, 4, 1, Resample::kBilinear);
  for (int x = 0; x < 4; ++x) EXPECT_EQ(at(half, x, 0), 20 * x + 5) << x;
  const auto flat = make_image(31, 17, [](int, int) { return 93; });
  for (unsigned char v : resize(flat, 13, 7, Resample::kBilinear).pixels) EXPECT_EQ(v, 93);
}

TEST(Pyramid, NearestKeepsMasksBinary) {
  const auto mask = make_image(100, 80, [](int x, int y) {
    return (x - 50) * (x - 50) + (y - 40) * (y - 40) < 900 ? 255 : 0;
  });
  for (const auto& level : build_pyramid(mask, 5, 1.2, Resample::kNearest)) {
    const std::set<unsigned char> values(level.pixels.begin(), level.pixels.end());
    EXPECT_EQ(values, (std::set<unsigned char>{0, 255}));
  }
}

TEST(Pyramid, GaussianBlurKeepsConstantsAndSmoothsSteps) {
  const auto flat = make_image(15, 15, [](int, int) { return 200; });
  EXPECT_EQ(gaussian_blur7(flat).pixels, flat.pixels);
  const auto step = make_image(15, 3, [](int x, int) { return x < 7 ? 0 : 200; });
  const auto b = gaussian_blur7(step);
  EXPECT_EQ(at(b, 0, 1), 0);
  EXPECT_EQ(at(b, 14, 1), 200);
  EXPECT_GT(at(b, 6, 1), 0);
  EXPECT_LT(at(b, 7, 1), 200);
  for (int x = 1; x < 15; ++x) EXPECT_GE(at(b, x, 1), at(b, x - 1, 1));  // monotone
}
