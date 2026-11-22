#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <Eigen/Geometry>

#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/pnp.hpp"

using namespace sfm::recon;  // NOLINT(google-build-using-namespace)

namespace {

double rot_deg(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  return Eigen::AngleAxisd(a * b.transpose()).angle() * 180.0 / M_PI;
}

struct View {
  sfm::eval::SyntheticScene s;
  std::vector<Eigen::Vector3d> X;
  std::vector<Eigen::Vector2d> uv;
};

View view(double half, double noise, std::uint64_t seed = 2) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 2;
  c.points = 150;
  c.box_half = half;
  c.noise_px = noise;
  c.focal_px = 1270;
  c.seed = seed;
  View v{sfm::eval::make_synthetic_scene(c), {}, {}};
  for (int p = 0; p < c.points; ++p) {
    v.X.push_back(v.s.points.col(p));
    v.uv.push_back(
        v.s.observations[static_cast<std::size_t>(c.points) + static_cast<std::size_t>(p)].uv);
  }
  return v;
}

}  // namespace

TEST(P3p, ExactOnEveryTripleOfANoiseFreeScene) {
  const View v = view(20.0, 0.0);
  const Pose& truth = v.s.poses[1];
  int exact = 0;
  int triples = 0;
  for (std::size_t k = 0; k + 2 < v.X.size(); k += 3) {
    std::array<Eigen::Vector3d, 3> X;
    std::array<Eigen::Vector3d, 3> b;
    for (std::size_t m = 0; m < 3; ++m) {
      X[m] = v.X[k + m];
      b[m] = normalise(v.s.K, v.uv[k + m]).homogeneous();
    }
    const auto sols = p3p(X, b);
    ASSERT_GE(sols.size(), 1u);
    ASSERT_LE(sols.size(), 4u);
    double best = 1e9;
    for (const Pose& p : sols) {
      best = std::min(best, rot_deg(p.R, truth.R) + (p.t - truth.t).norm());
    }
    // A triple near the "danger cylinder" (camera centre on the circle through the three points)
    // loses precision; measured: 49 of 50 triples below 1e-6, the worst 1.4e-4.
    EXPECT_LT(best, 1e-3) << k;
    exact += best < 1e-6 ? 1 : 0;
    ++triples;
  }
  EXPECT_GE(exact, triples - 2);
  const std::array<Eigen::Vector3d, 3> line = {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1),
                                               Eigen::Vector3d(2, 2, 2)};
  EXPECT_TRUE(
      p3p(line, {Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0.1, 0, 1), Eigen::Vector3d(0.2, 0, 1)})
          .empty());
}

TEST(Pnp, AbsoluteOrientationIsExact) {
  const Pose truth{exp_so3(Eigen::Vector3d(0.2, -0.4, 1.1)), Eigen::Vector3d(1, -2, 3)};
  std::vector<Eigen::Vector3d> p = {{0, 0, 0}, {1, 0, 0}, {0, 2, 0}, {0, 0, 3}, {1, 1, 1}};
  std::vector<Eigen::Vector3d> q;
  q.reserve(p.size());
  for (const auto& x : p) q.push_back(truth.R * x + truth.t);
  const Pose est = absolute_orientation(p, q);
  EXPECT_LT(rot_deg(est.R, truth.R), 1e-9);
  EXPECT_LT((est.t - truth.t).norm(), 1e-12);
}

TEST(Pnp, RansacRecoversThePoseWithOutliers) {
  for (const double half : {20.0, 5.0}) {  // wide and bunny-like narrow view
    View v = view(half, 0.5);
    sfm::eval::Rng rng(9);
    for (std::size_t k = 0; k < v.uv.size(); k += 3) {
      v.uv[k] = {rng.uniform(0, 1024), rng.uniform(0, 768)};
    }
    PnpParams p;
    p.threshold_px = 4.0;
    p.seed = 17;
    const PnpResult r = pnp_ransac(v.s.K, v.X, v.uv, p);
    ASSERT_TRUE(r.ok) << half;
    EXPECT_GE(r.inliers, 100);  // all 100 true correspondences (+ rare lucky outliers)
    EXPECT_LE(r.inliers, 103);
    EXPECT_LT(rot_deg(r.pose.R, v.s.poses[1].R), half > 10 ? 0.05 : 0.3) << half;
    const PnpResult again = pnp_ransac(v.s.K, v.X, v.uv, p);
    EXPECT_EQ(again.pose.R, r.pose.R);
    EXPECT_EQ(again.inlier, r.inlier);
  }
}

TEST(Pnp, NoiseFreeIsExactAndTooFewFail) {
  const View v = view(20.0, 0.0);
  const PnpResult r = pnp_ransac(v.s.K, v.X, v.uv, PnpParams{});
  ASSERT_TRUE(r.ok);
  EXPECT_EQ(r.inliers, 150);
  EXPECT_LT(rot_deg(r.pose.R, v.s.poses[1].R), 1e-8);
  EXPECT_LT((r.pose.t - v.s.poses[1].t).norm(), 1e-8);
  const std::vector<Eigen::Vector3d> X3(v.X.begin(), v.X.begin() + 3);
  const std::vector<Eigen::Vector2d> uv3(v.uv.begin(), v.uv.begin() + 3);
  EXPECT_FALSE(pnp_ransac(v.s.K, X3, uv3, PnpParams{}).ok);
}
