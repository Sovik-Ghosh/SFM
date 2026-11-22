#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/matching/five_point.hpp"

using sfm::matching::five_point;
using sfm::matching::ransac_essential;

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

struct Views {
  std::vector<Eigen::Vector2d> x1;
  std::vector<Eigen::Vector2d> x2;
  Eigen::Matrix3d E;  // unit norm
  double focal = 1.0;
};

// Normalised correspondences of cameras 0 and 1 of a synthetic scene; `planar` puts every point on
// the plane z = 0 (the case where the 8-point method is degenerate).
Views views(double noise_px, bool planar, std::uint64_t seed) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 2;
  c.points = 120;
  c.box_half = 15.0;
  c.noise_px = 0.0;
  c.seed = seed;
  auto s = sfm::eval::make_synthetic_scene(c);
  sfm::eval::Rng rng(seed + 100);
  Views v;
  v.focal = s.K(0, 0);
  const Eigen::Matrix3d Kinv = s.K.inverse();
  for (int p = 0; p < c.points; ++p) {
    Eigen::Vector3d X = s.points.col(p);
    if (planar) X.z() = 0.0;
    for (std::size_t k = 0; k < 2; ++k) {
      const auto& pose = s.poses[k];
      const Eigen::Vector3d h = s.K * (pose.R * X + pose.t);
      Eigen::Vector2d uv = h.head<2>() / h.z();
      uv += noise_px * Eigen::Vector2d(rng.normal(), rng.normal());
      const Eigen::Vector3d n = Kinv * uv.homogeneous();
      (k == 0 ? v.x1 : v.x2).push_back(n.head<2>() / n.z());
    }
  }
  const Eigen::Matrix3d R = s.poses[1].R * s.poses[0].R.transpose();
  const Eigen::Vector3d t = s.poses[1].t - R * s.poses[0].t;
  v.E = skew(t) * R;
  v.E /= v.E.norm();
  return v;
}

double e_distance(const Eigen::Matrix3d& a, const Eigen::Matrix3d& b) {
  return std::min((a - b).norm(), (a + b).norm());
}

}  // namespace

TEST(FivePoint, NoiseFreeOneSolutionIsExactGeneralAndPlanar) {
  for (const bool planar : {false, true}) {
    int exact = 0;
    int samples = 0;
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
      const Views v = views(0.0, planar, seed);
      Eigen::Vector2d a[5];
      Eigen::Vector2d b[5];
      for (std::size_t k = 0; k < 5; ++k) {
        a[k] = v.x1[7 * k];
        b[k] = v.x2[7 * k];
      }
      const auto sols = five_point(a, b);
      EXPECT_LE(sols.size(), 10u);
      double best = 1e9;
      for (const auto& E : sols) {
        best = std::min(best, e_distance(E, v.E));
        // Every solution is an essential matrix and fits the five points.
        const Eigen::Vector3d sv = Eigen::JacobiSVD<Eigen::Matrix3d>(E).singularValues();
        EXPECT_NEAR(sv(0), sv(1), 1e-6);
        EXPECT_LT(sv(2), 1e-6);
        for (std::size_t k = 0; k < 5; ++k) {
          EXPECT_LT(std::fabs(b[k].homogeneous().dot(E * a[k].homogeneous())), 1e-9);
        }
      }
      exact += best < 1e-6 ? 1 : 0;
      ++samples;
    }
    EXPECT_GE(exact, samples - 1) << (planar ? "planar" : "general");
  }
}

TEST(FivePoint, RansacRecoversTheEssentialMatrixOfAGeneralScene) {
  Views v = views(0.5, false, 9);
  sfm::matching::EssentialRansacParams p;
  p.threshold_px = 2.0;
  p.focal = v.focal;
  const auto r = ransac_essential(v.x1, v.x2, p);
  EXPECT_GE(r.inliers, 118);  // 0.5 px noise: a rare point exceeds the 2 px threshold
  EXPECT_LT(e_distance(r.E, v.E), 0.02);
}

TEST(FivePoint, RansacFindsInliersOnAPlaneWithOutliers) {
  Views v = views(0.5, true, 7);
  sfm::eval::Rng rng(3);
  std::vector<bool> outlier(v.x1.size(), false);
  for (std::size_t k = 0; k < v.x1.size(); k += 3) {
    v.x2[k] = Eigen::Vector2d(rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3));
    outlier[k] = true;
  }
  sfm::matching::EssentialRansacParams p;
  p.threshold_px = 2.0;
  p.focal = v.focal;
  p.seed = 11;
  const auto r = ransac_essential(v.x1, v.x2, p);
  int true_in = 0;
  int false_in = 0;
  for (std::size_t k = 0; k < v.x1.size(); ++k) {
    if (r.inlier[k] != 0) (outlier[k] ? false_in : true_in)++;
  }
  EXPECT_EQ(true_in, 80);
  EXPECT_LE(false_in, 3);
  // On a plane two essential matrices explain the points equally well (the planar twin), so the
  // chosen E need not be the true one; it must fit every true correspondence.
  for (std::size_t k = 0; k < v.x1.size(); ++k) {
    if (outlier[k]) continue;
    const Eigen::Vector3d Ea = r.E * v.x1[k].homogeneous();
    const double res = std::fabs(v.x2[k].homogeneous().dot(Ea)) / Ea.head<2>().norm();
    EXPECT_LT(res * v.focal, 3.0) << k;  // one-sided epipolar distance, pixels
  }
  const auto again = ransac_essential(v.x1, v.x2, p);
  EXPECT_EQ(again.E, r.E);
  EXPECT_EQ(again.inlier, r.inlier);
}
