#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/data/rig.hpp"

using sfm::data::parse_rig;

namespace {

const std::string kRig =
    "# comment\n"
    "SENSOR 0 cam4 1 0 0 0 1 0 0 0 1 0 0 0\n"
    "SENSOR 1 cam5 0 -1 0 1 0 0 0 0 1 0.06 0 0\n"
    "IMAGE 10 5 0\n"
    "IMAGE 11 5 1\n"
    "IMAGE 12 6 0\n"
    "IMAGE 13 6 1\n";

}  // namespace

TEST(Rig, ParsesSensorsAndRenumbersFramesOfTheLoadedImages) {
  const auto rig = parse_rig(kRig, {12, 13, 10}, "rig.txt");
  ASSERT_EQ(rig.sensors(), 2U);
  EXPECT_EQ(rig.frames, 2);
  EXPECT_EQ(rig.frame, (std::vector<int>{0, 0, 1}));  // frame 6 first appears first
  EXPECT_EQ(rig.sensor, (std::vector<int>{0, 1, 0}));
  EXPECT_EQ(rig.R[1](0, 1), -1.0);
  EXPECT_EQ(rig.t[1].x(), 0.06);
}

TEST(Rig, RejectsMissingImagesBadSensorsAndNonRotations) {
  EXPECT_THROW(parse_rig(kRig, {10, 14}, "rig.txt"), sfm::IoError);  // no IMAGE line for 14
  EXPECT_THROW(parse_rig(kRig + "IMAGE 14 7 2\n", {14}, "rig.txt"), sfm::IoError);  // sensor 2
  EXPECT_THROW(parse_rig("SENSOR 0 a 2 0 0 0 1 0 0 0 1 0 0 0\nIMAGE 0 0 0\n", {0}, "rig.txt"),
               sfm::IoError);
  EXPECT_THROW(parse_rig("SENSOR 1 a 1 0 0 0 1 0 0 0 1 0 0 0\nIMAGE 0 0 1\n", {0}, "rig.txt"),
               sfm::IoError);  // sensors must start at 0
  EXPECT_THROW(parse_rig("CAMERA 0\n", {}, "rig.txt"), sfm::IoError);
  EXPECT_THROW(sfm::data::load_rig("/nonexistent/rig.txt", {0}), sfm::IoError);
}
