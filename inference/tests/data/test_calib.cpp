#include <gtest/gtest.h>

#include <filesystem>

#include "sfm/core/error.hpp"
#include "sfm/data/calib.hpp"

#include "temp_dir.hpp"

using sfm::data::read_projection;
using Calib = sfm::data::test::TempDirTest;

TEST(CalibBunny, ReadsCamera0000) {
  const auto p =
      read_projection(std::filesystem::path(SFM_SOURCE_DIR) / "data/bunny/calib/0000.txt");
  EXPECT_EQ(p(0, 0), -729.446);
  EXPECT_EQ(p(0, 3), 30722.9);
  EXPECT_EQ(p(1, 0), 1038.24);
  EXPECT_EQ(p(2, 2), -0.999881);
  EXPECT_EQ(p(2, 3), 59.3402);
}

TEST_F(Calib, ParsesAnyWhitespaceAndCrlf) {
  const auto p = read_projection(write("a.txt", "CONTOUR\r\n1 2 3 4\n5 6 7 8 9\n10\t11 12\n\n"));
  for (int i = 0; i < 12; ++i) EXPECT_EQ(p(i / 4, i % 4), i + 1);
}

TEST_F(Calib, RejectsBadFiles) {
  EXPECT_THROW(read_projection(dir_ / "missing.txt"), sfm::IoError);
  EXPECT_THROW(read_projection(write("h.txt", "CONTOURS\n1 2 3 4 5 6 7 8 9 10 11 12\n")),
               sfm::IoError);
  EXPECT_THROW(read_projection(write("n11.txt", "CONTOUR\n1 2 3 4 5 6 7 8 9 10 11\n")),
               sfm::IoError);
  EXPECT_THROW(read_projection(write("n13.txt", "CONTOUR\n1 2 3 4 5 6 7 8 9 10 11 12 13\n")),
               sfm::IoError);
  EXPECT_THROW(read_projection(write("x.txt", "CONTOUR\n1 2 3 4 5 6 7 8 9 10 11 1x2\n")),
               sfm::IoError);
  EXPECT_THROW(read_projection(write("nan.txt", "CONTOUR\n1 2 3 4 5 6 7 8 9 10 11 nan\n")),
               sfm::IoError);
}
