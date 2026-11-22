#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/SVD>

#include "sfm/core/error.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/matching/fundamental.hpp"

#include "two_view.hpp"

using sfm::matching::eight_point;
using sfm::matching::Points2;
using sfm::matching::ransac_fundamental;
using sfm::matching::RansacParams;
using sfm::matching::sampson_sq;
using sfm::matching::test::f_distance;
using sfm::matching::test::fundamental_from_projections;

namespace {

struct TwoViews {
  Points2 x1;
  Points2 x2;
  Eigen::Matrix3d F;
};

// Cameras 0 and 1 of a synthetic scene (wide field: box_half 20), all points.
TwoViews two_views(double noise_px, std::uint64_t seed = 3) {
  sfm::eval::SyntheticConfig cfg;
  cfg.cameras = 2;
  cfg.points = 200;
  cfg.box_half = 20.0;
  cfg.noise_px = noise_px;
  cfg.seed = seed;
  const auto s = sfm::eval::make_synthetic_scene(cfg);
  TwoViews v;
  for (int p = 0; p < cfg.points; ++p) {
    v.x1.push_back(s.observations[static_cast<std::size_t>(p)].uv);
    v.x2.push_back(
        s.observations[static_cast<std::size_t>(cfg.points) + static_cast<std::size_t>(p)].uv);
  }
  Eigen::Matrix<double, 3, 4> P1;
  Eigen::Matrix<double, 3, 4> P2;
  P1 << s.K * s.poses[0].R, s.K * s.poses[0].t;
  P2 << s.K * s.poses[1].R, s.K * s.poses[1].t;
  v.F = fundamental_from_projections(P1, P2);
  return v;
}

}  // namespace

TEST(EightPoint, NoiseFreeRecoversF) {
  const auto v = two_views(0.0);
  const Eigen::Matrix3d F = eight_point(v.x1, v.x2);
  EXPECT_NEAR(F.norm(), 1.0, 1e-12);
  EXPECT_LT(f_distance(F, v.F), 1e-9);
  const Eigen::Vector3d s = Eigen::JacobiSVD<Eigen::Matrix3d>(F).singularValues();
  EXPECT_LT(s(2) / s(0), 1e-12);  // rank 2
  for (std::size_t k = 0; k < v.x1.size(); ++k) EXPECT_LT(sampson_sq(F, v.x1[k], v.x2[k]), 1e-12);
  // Exactly eight points are enough.
  const Points2 a(v.x1.begin(), v.x1.begin() + 8);
  const Points2 b(v.x2.begin(), v.x2.begin() + 8);
  EXPECT_LT(f_distance(eight_point(a, b), v.F), 1e-6);
}

TEST(EightPoint, NoisyIsCloseAndRejectsBadInput) {
  const auto v = two_views(0.5);
  const Eigen::Matrix3d F = eight_point(v.x1, v.x2);
  double mean = 0.0;
  for (std::size_t k = 0; k < v.x1.size(); ++k) mean += std::sqrt(sampson_sq(F, v.x1[k], v.x2[k]));
  mean /= static_cast<double>(v.x1.size());
  EXPECT_LT(mean, 0.5);  // fitted residual below the noise level
  const Points2 seven(v.x1.begin(), v.x1.begin() + 7);
  EXPECT_THROW(eight_point(seven, seven), sfm::GeometryError);
  EXPECT_THROW(eight_point(v.x1, seven), sfm::GeometryError);
  const Points2 same(8, Eigen::Vector2d(5, 5));
  EXPECT_THROW(eight_point(same, same), sfm::GeometryError);
}

TEST(Sampson, IsTheFirstOrderPixelDistance) {
  // F of a pure x-translation: epipolar lines are rows (v2 = v1); Sampson distance of a point
  // pair off by d rows is d / sqrt(2) (the error is split between the two images).
  Eigen::Matrix3d F;
  F << 0, 0, 0, 0, 0, -1, 0, 1, 0;
  EXPECT_NEAR(sampson_sq(F, {10, 20}, {50, 20}), 0.0, 1e-15);
  EXPECT_NEAR(std::sqrt(sampson_sq(F, {10, 20}, {50, 23})), 3.0 / std::sqrt(2.0), 1e-12);
}

TEST(RansacF, FindsTheInliersAmongOutliers) {
  auto v = two_views(0.3, 5);
  // Replace 40 % of the second-image points with uniform random pixels.
  sfm::eval::Rng rng(11);
  std::vector<bool> outlier(v.x1.size(), false);
  for (std::size_t k = 0; k < v.x1.size(); ++k) {
    if (k % 5 < 2) {
      v.x2[k] = {rng.uniform(0, 1024), rng.uniform(0, 768)};
      outlier[k] = true;
    }
  }
  RansacParams p;
  p.threshold_px = 2.0;
  p.seed = 42;
  const auto r = ransac_fundamental(v.x1, v.x2, p);
  int true_in = 0;
  int false_in = 0;
  for (std::size_t k = 0; k < v.x1.size(); ++k) {
    if (r.inlier[k] != 0) (outlier[k] ? false_in : true_in)++;
  }
  EXPECT_EQ(true_in, 120);  // every true inlier found
  EXPECT_LE(false_in, 3);   // random points can land near their epipolar line by chance
  EXPECT_EQ(r.inliers, true_in + false_in);
  EXPECT_LT(f_distance(r.F, v.F), 1e-2);
  EXPECT_LT(r.iterations, p.max_iters);  // adaptive stop

  const auto again = ransac_fundamental(v.x1, v.x2, p);
  EXPECT_EQ(again.inlier, r.inlier);
  EXPECT_EQ(again.F, r.F);  // bit-identical for the same seed
}

TEST(RansacF, TooFewPointsGiveNoModel) {
  const auto v = two_views(0.0);
  const Points2 a(v.x1.begin(), v.x1.begin() + 7);
  const Points2 b(v.x2.begin(), v.x2.begin() + 7);
  const auto r = ransac_fundamental(a, b, RansacParams{});
  EXPECT_EQ(r.inliers, 0);
  EXPECT_EQ(r.inlier.size(), 7u);
}
