// load_gray on bunny 0000 equals OpenCV's gray (tools/make_golden_p1.py) on every pixel.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "sfm/core/npy.hpp"
#include "sfm/data/image_io.hpp"

TEST(ImageIoGolden, BunnyGrayMatchesOpenCvExactly) {
  const std::filesystem::path src(SFM_SOURCE_DIR);
  const auto ref = sfm::core::npy_read(src / "inference/tests/golden/p1/gray_0000.npy");
  const auto img = sfm::data::load_gray(src / "data/bunny/images/0000.ppm");
  ASSERT_EQ(ref.shape, (std::vector<std::size_t>{768, 1024}));
  ASSERT_EQ(img.height, 768);
  ASSERT_EQ(img.width, 1024);

  const std::vector<std::uint8_t> expected = ref.to_vector<std::uint8_t>();
  std::size_t differ = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) differ += img.pixels[i] != expected[i];
  EXPECT_EQ(differ, 0u) << "of " << expected.size() << " pixels";
}
