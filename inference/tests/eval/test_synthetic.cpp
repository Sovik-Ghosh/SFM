#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <string>

#include <Eigen/LU>

#include "sfm/core/error.hpp"
#include "sfm/eval/synthetic.hpp"

using sfm::eval::make_synthetic_scene;
using sfm::eval::project;
using sfm::eval::SyntheticConfig;
using sfm::eval::SyntheticScene;

TEST(Synthetic, NoiseFreeObservationsReprojectExactly) {
  const SyntheticScene s = make_synthetic_scene(SyntheticConfig{});
  ASSERT_EQ(s.poses.size(), 12u);
  ASSERT_EQ(s.points.cols(), 400);
  ASSERT_EQ(s.observations.size(), 12u * 400u);
  for (const auto& o : s.observations) {
    EXPECT_FALSE(o.outlier);
    EXPECT_LT((o.uv - project(s.K, s.poses[o.camera], s.points.col(o.point))).norm(), 1e-9);
  }
}

TEST(Synthetic, CamerasAreRotationsOnTheHemisphereLookingAtTheOrigin) {
  const SyntheticConfig cfg;
  const SyntheticScene s = make_synthetic_scene(cfg);
  for (const auto& pose : s.poses) {
    EXPECT_LT((pose.R * pose.R.transpose() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    EXPECT_NEAR(pose.R.determinant(), 1.0, 1e-12);
    const Eigen::Vector3d c = pose.centre();
    EXPECT_NEAR(c.norm(), cfg.camera_distance, 1e-9);
    const double elevation = std::asin(c.z() / c.norm()) * 180.0 / 3.14159265358979323846;
    EXPECT_GE(elevation, cfg.min_elevation_deg - 1e-9);
    EXPECT_LE(elevation, cfg.max_elevation_deg + 1e-9);
    // The origin projects onto the principal point; world up (+z) points up in the image (-y).
    const Eigen::Vector2d o = project(s.K, pose, Eigen::Vector3d::Zero());
    EXPECT_LT((o - Eigen::Vector2d(512, 384)).norm(), 1e-9);
    EXPECT_LT(project(s.K, pose, Eigen::Vector3d(0, 0, 1)).y(), o.y());
  }
}

TEST(Synthetic, EveryPointIsInFrontOfEveryCameraAndInsideTheImage) {
  const SyntheticScene s = make_synthetic_scene(SyntheticConfig{});
  for (const auto& pose : s.poses) {
    for (Eigen::Index j = 0; j < s.points.cols(); ++j) {
      EXPECT_GT((pose.R * s.points.col(j) + pose.t).z(), 0.0);
      const Eigen::Vector2d uv = project(s.K, pose, s.points.col(j));
      EXPECT_TRUE(uv.x() >= 0 && uv.x() < 1024 && uv.y() >= 0 && uv.y() < 768) << uv.transpose();
    }
  }
}

TEST(Synthetic, OutlierFractionIsHonouredExactly) {
  SyntheticConfig cfg;
  cfg.outlier_fraction = 0.1;  // 4800 observations -> exactly 480 outliers
  const SyntheticScene s = make_synthetic_scene(cfg);
  std::size_t outliers = 0;
  for (const auto& o : s.observations) {
    outliers += o.outlier;
    if (!o.outlier) {
      EXPECT_LT((o.uv - project(s.K, s.poses[o.camera], s.points.col(o.point))).norm(), 1e-9);
    }
  }
  EXPECT_EQ(outliers, 480u);
}

TEST(Synthetic, NoiseHasTheRequestedSigma) {
  SyntheticConfig cfg;
  cfg.noise_px = 0.5;
  const SyntheticScene s = make_synthetic_scene(cfg);
  double sum_sq = 0.0;
  for (const auto& o : s.observations) {
    sum_sq += (o.uv - project(s.K, s.poses[o.camera], s.points.col(o.point))).squaredNorm();
  }
  const double sigma = std::sqrt(sum_sq / (2.0 * static_cast<double>(s.observations.size())));
  EXPECT_NEAR(sigma, 0.5, 0.02);  // 9600 samples: standard error ~0.004
}

TEST(Synthetic, SameSeedSameSceneDifferentSeedDifferentScene) {
  SyntheticConfig cfg;
  cfg.noise_px = 1.0;
  cfg.outlier_fraction = 0.05;
  const SyntheticScene a = make_synthetic_scene(cfg);
  const SyntheticScene b = make_synthetic_scene(cfg);
  EXPECT_EQ(a.points, b.points);
  for (std::size_t i = 0; i < a.observations.size(); ++i) {
    ASSERT_EQ(a.observations[i].uv, b.observations[i].uv);
    ASSERT_EQ(a.observations[i].outlier, b.observations[i].outlier);
  }
  cfg.seed = 2;
  EXPECT_NE(make_synthetic_scene(cfg).points, a.points);
}

TEST(Synthetic, InvalidConfigNamesTheField) {
  auto message = [](SyntheticConfig cfg) {
    try {
      make_synthetic_scene(cfg);
    } catch (const sfm::ConfigError& e) {
      return std::string(e.what());
    }
    return std::string("no error");
  };
  SyntheticConfig c;
  c.cameras = 0;
  EXPECT_NE(message(c).find("synthetic.cameras"), std::string::npos);
  c = {};
  c.box_half = 40.0;  // 40 * sqrt(3) > 60: a point could be behind a camera
  EXPECT_NE(message(c).find("synthetic.box_half"), std::string::npos);
  c = {};
  c.max_elevation_deg = 90.0;
  EXPECT_NE(message(c).find("synthetic.max_elevation_deg"), std::string::npos);
  c = {};
  c.outlier_fraction = 1.5;
  EXPECT_NE(message(c).find("synthetic.outlier_fraction"), std::string::npos);
}
