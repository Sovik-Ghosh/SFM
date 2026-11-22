#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/geometry.hpp"

using namespace sfm::recon;  // NOLINT(google-build-using-namespace)

namespace {

double rot_deg(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  return Eigen::AngleAxisd(a * b.transpose()).angle() * 180.0 / M_PI;
}

sfm::eval::SyntheticScene scene(int cameras, double noise, double half = 20.0) {
  sfm::eval::SyntheticConfig c;
  c.cameras = cameras;
  c.points = 100;
  c.box_half = half;
  c.noise_px = noise;
  return sfm::eval::make_synthetic_scene(c);
}

// Pose of camera b relative to camera a (camera a becomes [I | 0]).
Pose relative(const Pose& a, const Pose& b) {
  const Eigen::Matrix3d R = b.R * a.R.transpose();
  return {R, b.t - R * a.t};
}

}  // namespace

TEST(Geometry, NearestEssentialHasEqualSingularValues) {
  sfm::eval::Rng rng(3);
  Eigen::Matrix3d M;
  for (int k = 0; k < 9; ++k) M(k / 3, k % 3) = rng.normal();
  const Eigen::Vector3d s =
      Eigen::JacobiSVD<Eigen::Matrix3d>(nearest_essential(M)).singularValues();
  EXPECT_NEAR(s(0), 1.0, 1e-12);
  EXPECT_NEAR(s(1), 1.0, 1e-12);
  EXPECT_NEAR(s(2), 0.0, 1e-12);
}

TEST(Geometry, EssentialDecompositionAndCheirality) {
  const auto s = scene(2, 0.0);
  const Pose rel = relative(s.poses[0], s.poses[1]);
  const Eigen::Matrix3d E = skew(rel.t.normalized()) * rel.R;
  // E from the exact F: F = K^-T E K^-1.
  const Eigen::Matrix3d Kinv = s.K.inverse();
  const Eigen::Matrix3d F = Kinv.transpose() * E * Kinv;
  const Eigen::Matrix3d E2 = essential_from_fundamental(F, s.K, s.K);
  EXPECT_LT(std::min((E2 - E).norm(), (E2 + E).norm()), 1e-9);

  const auto cand = decompose_essential(E);
  int matching = 0;
  for (const Pose& p : cand) {
    EXPECT_NEAR(p.R.determinant(), 1.0, 1e-12);
    EXPECT_NEAR(p.t.norm(), 1.0, 1e-12);
    if (rot_deg(p.R, rel.R) < 1e-6 && (p.t - rel.t.normalized()).norm() < 1e-9) ++matching;
  }
  EXPECT_EQ(matching, 1);

  std::vector<Eigen::Vector2d> x1;
  std::vector<Eigen::Vector2d> x2;
  for (int p = 0; p < 100; ++p) {
    x1.push_back(normalise(s.K, s.observations[static_cast<std::size_t>(p)].uv));
    x2.push_back(normalise(s.K, s.observations[100 + static_cast<std::size_t>(p)].uv));
  }
  int in_front = 0;
  const Pose chosen = select_relative_pose(E, x1, x2, in_front);
  EXPECT_EQ(in_front, 100);
  EXPECT_LT(rot_deg(chosen.R, rel.R), 1e-6);
  EXPECT_LT((chosen.t - rel.t.normalized()).norm(), 1e-9);
}

TEST(Geometry, TriangulateDltExactAndBounded) {
  const auto s = scene(4, 0.0);
  for (int p = 0; p < 100; ++p) {
    std::vector<Mat34> P;
    std::vector<Eigen::Vector2d> x;
    for (int c = 0; c < 4; ++c) {
      P.push_back(pose_matrix(s.poses[static_cast<std::size_t>(c)]));
      x.push_back(normalise(
          s.K, s.observations[static_cast<std::size_t>(c) * 100 + static_cast<std::size_t>(p)].uv));
    }
    Eigen::Vector3d X;
    ASSERT_TRUE(triangulate_dlt(P, x, X));  // n-view
    EXPECT_LT((X - s.points.col(p)).norm(), 1e-9);
    ASSERT_TRUE(triangulate_dlt({P[0], P[2]}, {x[0], x[2]}, X));  // two-view
    EXPECT_LT((X - s.points.col(p)).norm(), 1e-9);
  }
  // Noisy: error bounded by the noise times the depth / (focal * baseline factor).
  const auto n = scene(4, 0.5);
  double worst = 0.0;
  for (int p = 0; p < 100; ++p) {
    std::vector<Mat34> P;
    std::vector<Eigen::Vector2d> x;
    for (int c = 0; c < 4; ++c) {
      P.push_back(pose_matrix(n.poses[static_cast<std::size_t>(c)]));
      x.push_back(normalise(
          n.K, n.observations[static_cast<std::size_t>(c) * 100 + static_cast<std::size_t>(p)].uv));
    }
    Eigen::Vector3d X;
    ASSERT_TRUE(triangulate_dlt(P, x, X));
    worst = std::max(worst, (X - n.points.col(p)).norm());
  }
  EXPECT_LT(worst, 0.5);  // 60 units away, 1200 px focal: 0.5 px ~ 0.025 units laterally
  Eigen::Vector3d X;
  EXPECT_FALSE(triangulate_dlt({Mat34::Identity()}, {Eigen::Vector2d(0, 0)}, X));
  // Parallel rays from two different centres (same image point, translated camera): the only
  // solution is the point at infinity in that direction.
  Mat34 P2 = Mat34::Identity();
  P2(0, 3) = -1.0;  // centre at x = 1
  const Eigen::Vector2d x(0.05, 0.02);
  EXPECT_FALSE(triangulate_dlt({Mat34::Identity(), P2}, {x, x}, X));
}

TEST(Geometry, HelpersAgree) {
  const auto s = scene(2, 0.0);
  const Pose& p = s.poses[1];
  const Eigen::Vector3d X = s.points.col(7);
  EXPECT_LT((project_px(s.K, p, X) - sfm::eval::project(s.K, p, X)).norm(), 1e-9);
  EXPECT_LT(reprojection_error(s.K, p, X, s.observations[107].uv), 1e-9);
  EXPECT_GT(depth(p, X), 0.0);
  const Eigen::Vector2d n = normalise(s.K, project_px(s.K, p, X));
  const Eigen::Vector3d xc = p.R * X + p.t;
  EXPECT_LT((n - xc.head<2>() / xc.z()).norm(), 1e-12);
  const Eigen::Vector3d w(0.3, -0.2, 0.9);
  EXPECT_LT((log_so3(exp_so3(w)) - w).norm(), 1e-12);
  EXPECT_NEAR(ray_angle_deg({-1, 0, 0}, {1, 0, 0}, {0, 0, 1}), 90.0, 1e-12);
  EXPECT_NEAR(nearest_rotation(2.0 * exp_so3(w)).determinant(), 1.0, 1e-12);
}
