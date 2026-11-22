#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/eval/synthetic.hpp"

using sfm::eval::Pose;
using sfm::eval::pose_error;

namespace {

std::vector<Pose> truth() { return sfm::eval::make_synthetic_scene({}).poses; }

// The same cameras seen in another frame: X' = s R X + t. A camera with pose (R_i, C_i) becomes
// (R_i R^T, s R C_i + t).
std::vector<Pose> in_other_frame(const std::vector<Pose>& poses, double s, const Eigen::Matrix3d& R,
                                 const Eigen::Vector3d& t) {
  std::vector<Pose> out;
  out.reserve(poses.size());
  for (const auto& p : poses) {
    out.push_back(Pose::from_centre(p.R * R.transpose(), s * R * p.centre() + t));
  }
  return out;
}

}  // namespace

TEST(PoseError, TruthAgainstItselfIsZero) {
  const auto gt = truth();
  const auto e = pose_error(gt, gt);
  ASSERT_EQ(e.rotation_deg.size(), gt.size());
  EXPECT_LT(e.max_rotation_deg(), 1e-6);
  EXPECT_LT(e.max_centre_pct(), 1e-9);
  EXPECT_NEAR(e.alignment.s, 1.0, 1e-12);
}

TEST(PoseError, SimilarityTransformedTruthIsZero) {
  const auto gt = truth();
  const Eigen::Matrix3d R =
      Eigen::AngleAxisd(0.8, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  const auto est = in_other_frame(gt, 0.04, R, Eigen::Vector3d(10, -3, 7));  // as an SfM run would
  const auto e = pose_error(est, gt);
  EXPECT_LT(e.max_rotation_deg(), 1e-6);
  EXPECT_LT(e.max_centre_pct(), 1e-9);
  EXPECT_NEAR(e.alignment.s, 1.0 / 0.04, 1e-9);
}

TEST(PoseError, OneDegreeRotationErrorIsReportedAsOneDegree) {
  const auto gt = truth();
  auto est = gt;
  const Eigen::Matrix3d tilt = Eigen::AngleAxisd(1.0 * 3.14159265358979323846 / 180.0,
                                                 Eigen::Vector3d(0.3, -0.4, 0.5).normalized())
                                   .toRotationMatrix();
  est[4] = Pose::from_centre(tilt * gt[4].R, gt[4].centre());  // same centre, rotated by 1 degree
  const auto e = pose_error(est, gt);
  EXPECT_NEAR(e.rotation_deg[4], 1.0, 1e-6);
  for (std::size_t i = 0; i < gt.size(); ++i) {
    if (i != 4) {
      EXPECT_LT(e.rotation_deg[i], 1e-6) << i;
    }
  }
  EXPECT_LT(e.max_centre_pct(), 1e-9);
}

TEST(PoseError, CentreErrorIsAPercentageOfTheSceneRadius) {
  const auto gt = truth();
  auto est = gt;
  est[0] = Pose::from_centre(gt[0].R, gt[0].centre() + Eigen::Vector3d(0, 0, 3));
  const auto e = pose_error(est, gt);
  EXPECT_GT(e.centre_pct[0], 0.0);
  EXPECT_GT(e.scene_radius, 0.0);
  // The alignment spreads the error over all cameras, but camera 0 keeps the largest share.
  EXPECT_EQ(e.max_centre_pct(), e.centre_pct[0]);
  EXPECT_LT(e.centre_pct[0], 100.0 * 3.0 / e.scene_radius + 1e-9);
}

TEST(PoseError, MismatchedOrTooFewCamerasThrow) {
  const auto gt = truth();
  EXPECT_THROW(pose_error(std::vector<Pose>(gt.begin(), gt.end() - 1), gt), sfm::GeometryError);
  const std::vector<Pose> two(gt.begin(), gt.begin() + 2);
  EXPECT_THROW(pose_error(two, two), sfm::GeometryError);
}
