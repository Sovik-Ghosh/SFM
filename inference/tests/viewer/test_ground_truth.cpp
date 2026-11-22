#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "sfm/eval/pose_error.hpp"
#include "sfm/recon/geometry.hpp"
#include "sfm/viewer/ground_truth.hpp"

#include "snapshot_util.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(GroundTruth, KnownSimilarityIsUndoneExactly) {
  // Estimate = truth seen in another frame: X_est = s R X_true + t.
  const auto truth_snap = test::synthetic_snapshot(6, 10);
  const double s = 0.37;
  const Eigen::Matrix3d R = sfm::recon::exp_so3(Eigen::Vector3d(0.3, -0.5, 1.2));
  const Eigen::Vector3d t(4, -1, 2);
  auto est = truth_snap;
  std::vector<sfm::eval::Pose> truth;
  for (std::size_t i = 0; i < 6; ++i) {
    truth.push_back({truth_snap.R[i], truth_snap.t[i]});
    // x = R_t X_true + t_t, X_true = R^T (X_est - t) / s  ->  x s = R_t R^T X_est + (s t_t - R_t
    // R^T t)
    est.R[i] = truth_snap.R[i] * R.transpose();
    est.t[i] = s * truth_snap.t[i] - est.R[i] * t;
  }
  const GroundTruthView g = align_ground_truth(est, truth);
  ASSERT_TRUE(g.valid);
  for (std::size_t i = 0; i < 6; ++i) {
    // Aligned truth equals the estimate (same camera, same frame).
    EXPECT_LT((g.aligned[i].R - est.R[i]).norm(), 1e-9) << i;
    EXPECT_LT((g.aligned[i].centre() - sfm::eval::Pose{est.R[i], est.t[i]}.centre()).norm(), 1e-9)
        << i;
    EXPECT_LT(g.rotation_deg[i], 1e-6);
    EXPECT_LT(g.centre_pct[i], 1e-6);
  }
}

TEST(GroundTruth, ErrorsAreThoseOfPoseError) {
  auto est = test::synthetic_snapshot(6, 10, 2);
  std::vector<sfm::eval::Pose> truth;
  for (std::size_t i = 0; i < 6; ++i) truth.push_back({est.R[i], est.t[i]});
  est.R[3] = sfm::recon::exp_so3(Eigen::Vector3d(0.02, 0.0, 0.01)) * est.R[3];  // perturb one
  est.t[3] += Eigen::Vector3d(0.5, 0, 0);
  est.registered[5] = 0;
  const GroundTruthView g = align_ground_truth(est, truth);
  std::vector<sfm::eval::Pose> e;
  std::vector<sfm::eval::Pose> tr;
  for (std::size_t i = 0; i < 5; ++i) {
    e.push_back({est.R[i], est.t[i]});
    tr.push_back(truth[i]);
  }
  const auto ref = sfm::eval::pose_error(e, tr);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(g.rotation_deg[i], ref.rotation_deg[i]);
    EXPECT_DOUBLE_EQ(g.centre_pct[i], ref.centre_pct[i]);
  }
  EXPECT_EQ(g.rotation_deg[5], -1.0);  // not registered

  auto two = est;
  for (std::size_t i = 2; i < 6; ++i) two.registered[i] = 0;
  EXPECT_FALSE(align_ground_truth(two, truth).valid);
}
