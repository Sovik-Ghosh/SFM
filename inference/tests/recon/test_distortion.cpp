#include <gtest/gtest.h>

#include <vector>

#include <Eigen/Core>

#include "sfm/recon/distortion.hpp"

using sfm::recon::Radial;

namespace {

Eigen::Matrix3d camera() {
  Eigen::Matrix3d K;
  K << 850.0, 0.3, 775.0, 0.0, 852.0, 516.0, 0.0, 0.0, 1.0;
  return K;
}

}  // namespace

TEST(Distortion, UndistortInvertsDistortAcrossTheImage) {
  const Eigen::Matrix3d K = camera();
  for (const Radial d : {Radial{-0.1, 0.0}, Radial{0.05, -0.01}, Radial{-0.25, 0.05}}) {
    for (double u = 0.0; u <= 1550.0; u += 155.0) {
      for (double v = 0.0; v <= 1032.0; v += 129.0) {
        const Eigen::Vector2d p(u, v);
        const Eigen::Vector2d back = sfm::recon::undistort(K, d, sfm::recon::distort(K, d, p));
        EXPECT_LT((back - p).norm(), 1e-6) << d.k1 << " " << u << " " << v;
      }
    }
  }
}

TEST(Distortion, MatchesTheRadialFormula) {
  const Eigen::Matrix3d K = Eigen::Vector3d(1000.0, 1000.0, 1.0).asDiagonal();
  // Normalised (0.5, 0.25): r^2 = 0.3125; factor 1 + k1 r^2 + k2 r^4.
  const Radial d{-0.1, 0.02};
  const double f = 1.0 - 0.1 * 0.3125 + 0.02 * 0.3125 * 0.3125;
  const Eigen::Vector2d out = sfm::recon::distort(K, d, {500.0, 250.0});
  EXPECT_NEAR(out.x(), 500.0 * f, 1e-9);
  EXPECT_NEAR(out.y(), 250.0 * f, 1e-9);
  EXPECT_FALSE(Radial{}.any());
  EXPECT_TRUE(d.any());
}

TEST(Distortion, UndistortFeaturesMovesOnlyPositions) {
  sfm::features::Features f;
  f.xy = {775.0F, 516.0F, 1500.0F, 1000.0F};
  f.angle = {1.0F, 2.0F};
  const std::vector<Eigen::Matrix3d> K{camera()};
  const auto none = sfm::recon::undistort_features({f}, K, Radial{});
  EXPECT_EQ(none[0].xy, f.xy);
  const auto u = sfm::recon::undistort_features({f}, K, Radial{-0.1, 0.0});
  EXPECT_NEAR(u[0].xy[0], 775.0F, 1e-3);  // the principal point does not move
  EXPECT_GT(u[0].xy[2], 1500.0F);         // barrel: the corner moves outwards when undistorted
  EXPECT_EQ(u[0].angle, f.angle);
}
