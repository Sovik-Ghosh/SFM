#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/data/image_io.hpp"

#include "temp_dir.hpp"

using sfm::data::load_gray;
using sfm::data::load_mask;
using sfm::data::rgb_to_gray;
using ImageIo = sfm::data::test::TempDirTest;

TEST(RgbToGray, MatchesOpenCvValues) {
  // Expected values from cv2.cvtColor(..., COLOR_RGB2GRAY), OpenCV 4.14.
  EXPECT_EQ(rgb_to_gray(0, 0, 0), 0);
  EXPECT_EQ(rgb_to_gray(255, 255, 255), 255);
  EXPECT_EQ(rgb_to_gray(255, 0, 0), 76);
  EXPECT_EQ(rgb_to_gray(0, 255, 0), 150);
  EXPECT_EQ(rgb_to_gray(0, 0, 255), 29);
  EXPECT_EQ(rgb_to_gray(17, 200, 99), 134);
}

TEST_F(ImageIo, ColourImageBecomesGrayRowMajor) {
  // 2x2: red, green / blue, white  (file order is row-major RGB)
  const std::string px =
      std::string("\xff\x00\x00\x00\xff\x00", 6) + std::string("\x00\x00\xff\xff\xff\xff", 6);
  const auto img = load_gray(write_pnm("c.ppm", "P6", 2, 2, px));
  EXPECT_EQ(img.width, 2);
  EXPECT_EQ(img.height, 2);
  EXPECT_EQ(img.pixels, (std::vector<std::uint8_t>{76, 150, 29, 255}));
}

TEST_F(ImageIo, NonSquareKeepsWidthAndHeight) {
  const auto img = load_gray(write_pnm("w.pgm", "P5", 3, 1, std::string("\x01\x02\x03", 3)));
  EXPECT_EQ(img.width, 3);
  EXPECT_EQ(img.height, 1);
  EXPECT_EQ(img.pixels, (std::vector<std::uint8_t>{1, 2, 3}));  // gray file passes through
}

TEST_F(ImageIo, MaskIsObjectWhereSilhouetteIsZero) {
  // Non-binary silhouette like the bunny set: only exact 0 is object.
  const std::string px("\x00\x01\x10\xff", 4);
  const auto m = load_mask(write_pnm("s.pgm", "P5", 2, 2, px));
  EXPECT_EQ(m.pixels, (std::vector<std::uint8_t>{255, 0, 0, 0}));
}

TEST_F(ImageIo, MissingOrCorruptFilesThrow) {
  EXPECT_THROW(load_gray(dir_ / "missing.ppm"), sfm::IoError);
  EXPECT_THROW(load_mask(dir_ / "missing.pgm"), sfm::IoError);
  EXPECT_THROW(load_gray(write("junk.ppm", "not an image")), sfm::IoError);
  EXPECT_THROW(load_gray(write_pnm("short.ppm", "P6", 4, 4, "abc")), sfm::IoError);
}
