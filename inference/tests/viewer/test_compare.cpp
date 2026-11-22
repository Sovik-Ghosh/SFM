#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "sfm/viewer/compare.hpp"

#include "snapshot_util.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)
using sfm::pipeline::SceneSnapshot;

namespace {

sfm::eval::Similarity some_similarity() {
  sfm::eval::Similarity s;
  s.s = 2.5;
  s.R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, -2, 0.5).normalized()).toRotationMatrix();
  s.t = Eigen::Vector3d(3, -1, 7);
  return s;
}

Eigen::Vector2d project(const SceneSnapshot& s, std::size_t cam, const Eigen::Vector3d& p) {
  const Eigen::Vector3d x = s.K[cam] * (s.R[cam] * p + s.t[cam]);
  return x.head<2>() / x.z();
}

}  // namespace

TEST(Compare, TransformKeepsEveryImageAndAlignmentRecoversTheSimilarity) {
  const SceneSnapshot a = test::synthetic_snapshot(6, 40);
  const sfm::eval::Similarity sim = some_similarity();
  const SceneSnapshot moved = transform_snapshot(a, sim);
  // Same images: every point projects to the same pixel in every camera.
  for (std::size_t c = 0; c < a.R.size(); ++c) {
    EXPECT_NEAR((moved.R[c] * moved.R[c].transpose() - Eigen::Matrix3d::Identity()).norm(), 0,
                1e-12);
    for (std::size_t p = 0; p < a.points.size(); p += 7) {
      EXPECT_LT((project(moved, c, moved.points[p]) - project(a, c, a.points[p])).norm(), 1e-8);
    }
  }
  // B = A in another frame; align B onto A gives the inverse similarity and B lands on A.
  const SnapshotAlignment al = align_snapshots(moved, a);
  ASSERT_TRUE(al.valid);
  EXPECT_EQ(al.common, 6);
  EXPECT_NEAR(al.sim.s, 1.0 / sim.s, 1e-9);
  const SceneSnapshot back = transform_snapshot(moved, al.sim);
  for (std::size_t p = 0; p < a.points.size(); ++p) {
    EXPECT_LT((back.points[p] - a.points[p]).norm(), 1e-8);
  }
  for (std::size_t c = 0; c < a.R.size(); ++c) {
    EXPECT_LT((back.R[c] - a.R[c]).norm(), 1e-9);
    EXPECT_LT((back.t[c] - a.t[c]).norm(), 1e-8);
  }
}

TEST(Compare, AlignmentMatchesImagesByDatasetIdAndNeedsThree) {
  const SceneSnapshot a = test::synthetic_snapshot(6, 10);
  SceneSnapshot b = transform_snapshot(a, some_similarity());
  // B holds the images in another order and misses one registration.
  std::swap(b.ids[0], b.ids[5]);
  std::swap(b.R[0], b.R[5]);
  std::swap(b.t[0], b.t[5]);
  b.registered[2] = 0;
  const SnapshotAlignment al = align_snapshots(b, a);
  ASSERT_TRUE(al.valid);
  EXPECT_EQ(al.common, 5);
  EXPECT_NEAR(al.sim.s, 1.0 / 2.5, 1e-9);
  for (std::size_t i = 2; i < b.registered.size(); ++i) b.registered[i] = 0;
  EXPECT_FALSE(align_snapshots(b, a).valid);  // 2 in common
  EXPECT_EQ(align_snapshots(b, a).common, 2);
}

TEST(Compare, ConfigDiffListsChangedLeavesOnly) {
  const std::string a = R"({"run":{"name":"a","output_dir":"x","seed":42},
    "features":{"sift":{"peak_threshold":0.0067,"octave_resolution":3},"max_features":8192},
    "matching":{"ratio":0.8}})";
  const std::string b = R"({"run":{"name":"b","output_dir":"y","seed":42},
    "features":{"sift":{"peak_threshold":0.002,"octave_resolution":4},"max_features":8192},
    "matching":{"ratio":0.8,"mutual":true}})";
  const auto d = config_diff(a, b);
  ASSERT_EQ(d.size(), 3U);
  EXPECT_EQ(d[0].key, "features.sift.octave_resolution");
  EXPECT_EQ(d[0].a, "3");
  EXPECT_EQ(d[0].b, "4");
  EXPECT_EQ(d[1].key, "features.sift.peak_threshold");
  EXPECT_EQ(d[2].key, "matching.mutual");
  EXPECT_EQ(d[2].a, "");
  EXPECT_EQ(d[2].b, "true");
  EXPECT_TRUE(config_diff(a, a).empty());
}

TEST(Compare, MetricRowsKnowWhichDirectionIsBetter) {
  sfm::pipeline::Metrics a;
  sfm::pipeline::Metrics b;
  a.registered_ids = {0, 1, 2};
  b.registered_ids = {0, 1, 2};
  a.points = 3736;
  b.points = 5948;
  a.reproj_median_px = 0.230;
  b.reproj_median_px = 0.238;
  a.rotation_median_deg = 0.258;
  b.rotation_median_deg = 0.200;
  a.rotation_max_deg = 0.575;
  b.rotation_max_deg = 0.547;
  a.centre_median_pct = b.centre_median_pct = 0.4;
  a.centre_max_pct = b.centre_max_pct = 1.0;
  const auto rows = metric_rows(a, b, 0.89, 0.98);
  auto find = [&](const std::string& n) {
    for (const auto& r : rows) {
      if (r.name == n) return r;
    }
    return MetricRow{};
  };
  EXPECT_EQ(verdict(find("Registered")), 0);
  EXPECT_EQ(verdict(find("Points")), 1);            // more is better
  EXPECT_EQ(verdict(find("Reprojection px")), -1);  // higher error is worse
  EXPECT_EQ(verdict(find("Rotation median °")), 1);
  EXPECT_EQ(verdict(find("Total s")), -1);
  EXPECT_EQ(verdict(find("BA runs")), 0);
  a.rotation_median_deg = -1;  // no ground truth on one side: no verdict on the error rows
  const auto no_gt = metric_rows(a, b, 0.89, 0.98);
  for (const auto& r : no_gt) {
    if (r.name.rfind("Rotation", 0) == 0) {
      EXPECT_EQ(verdict(r), 0);
    }
  }
}

TEST(Compare, MetricsJsonRoundTrip) {
  sfm::pipeline::Metrics m;
  m.images = 36;
  m.keypoints_mean = 1236.25;
  m.verified_pairs = 251;
  m.tracks = 5972;
  m.registered_ids = {0, 3, 7};
  m.points = 5948;
  m.observations = 23708;
  m.reproj_median_px = 0.238;
  m.reproj_mean_px = 0.333;
  m.reproj_p95_px = 0.963;
  m.rotation_median_deg = 0.2;
  m.rotation_max_deg = 0.547;
  m.centre_median_pct = 0.348;
  m.centre_max_pct = 0.935;
  m.ba_runs = 9;
  EXPECT_EQ(sfm::pipeline::to_json(parse_metrics(sfm::pipeline::to_json(m))),
            sfm::pipeline::to_json(m));
  sfm::pipeline::Timings t;
  t.features = 0.588;
  t.ba = 0.271;
  t.total = 1.02;
  EXPECT_EQ(sfm::pipeline::to_json(parse_timings(sfm::pipeline::to_json(t))),
            sfm::pipeline::to_json(t));
}
