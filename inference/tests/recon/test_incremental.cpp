#include <gtest/gtest.h>

#include <algorithm>

#include "sfm/core/error.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/recon/incremental.hpp"

#include "scene.hpp"

using namespace sfm::recon;  // NOLINT(google-build-using-namespace)
using sfm::recon::test::make_input;

namespace {

sfm::eval::SyntheticConfig config(double noise) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 8;
  c.points = 200;
  c.box_half = 20.0;
  c.noise_px = noise;
  return c;
}

}  // namespace

TEST(TrackIndex, ByTrackAndByImage) {
  const auto in = make_input(config(0.0));
  const TrackIndex idx(in.tracks, in.features);
  EXPECT_EQ(idx.track_count(), 200);
  EXPECT_EQ(idx.image_count(), 8);
  EXPECT_EQ(idx.end(5) - idx.begin(5), 8);
  EXPECT_EQ(idx.begin(5)->image, 0);
  EXPECT_EQ(idx.in_image(3).size(), 200u);
  const auto& f = in.features[2];
  const TrackObs& o = *(idx.begin(9) + 2);
  EXPECT_EQ(o.image, 2);
  EXPECT_EQ(o.uv.x(), static_cast<double>(f.xy[2 * static_cast<std::size_t>(o.keypoint)]));
}

TEST(Incremental, NoiseFreeSceneIsRecoveredExactlyWithoutBa) {
  const auto in = make_input(config(0.0));
  const TrackIndex idx(in.tracks, in.features);
  IncrementalParams p;
  IncrementalReport rep;
  const Reconstruction rec =
      reconstruct_incremental(in.pairs, in.features, idx, in.K, p, {}, nullptr, &rep);
  EXPECT_EQ(rec.registered_count(), 8);
  EXPECT_EQ(rec.point_count(), 200);
  EXPECT_EQ(rep.pnp_failures, 0);
  EXPECT_EQ(rep.ba_runs, 0);
  std::vector<sfm::eval::Pose> est;
  est.reserve(rec.cameras.size());
  for (const Camera& c : rec.cameras) est.push_back(c.pose);
  const auto e = sfm::eval::pose_error(est, in.scene.poses);
  EXPECT_LT(e.max_rotation_deg(), 1e-5);
  EXPECT_LT(e.max_centre_pct(), 1e-5);
  const auto r = reprojection_errors(rec, idx);
  EXPECT_LT(*std::max_element(r.begin(), r.end()), 1e-3);  // float keypoints: ~1e-5 px
}

TEST(Incremental, CallsBaAfterInitPeriodicallyAndAtTheEnd) {
  const auto in = make_input(config(0.3));
  const TrackIndex idx(in.tracks, in.features);
  IncrementalParams p;
  p.ba_every_n_cameras = 3;
  std::vector<int> cameras_at_ba;
  const BundleAdjuster ba = [&](Reconstruction& r, const std::vector<int>& free_images) {
    EXPECT_TRUE(free_images.empty());
    cameras_at_ba.push_back(r.registered_count());
  };
  IncrementalReport rep;
  const Reconstruction rec =
      reconstruct_incremental(in.pairs, in.features, idx, in.K, p, ba, nullptr, &rep);
  EXPECT_EQ(rec.registered_count(), 8);
  // init (2), +3 -> 5, +3 -> 8, then nothing left and nothing new since the last BA.
  EXPECT_EQ(cameras_at_ba, (std::vector<int>{2, 5, 8}));
  EXPECT_EQ(rep.ba_runs, 3);
}

TEST(Incremental, FilterPointsAndNoInitialPair) {
  const auto in = make_input(config(0.0));
  const TrackIndex idx(in.tracks, in.features);
  IncrementalParams p;
  Reconstruction rec = reconstruct_incremental(in.pairs, in.features, idx, in.K, p, {}, nullptr);
  rec.points[3] += Eigen::Vector3d(0.0, 0.0, 5.0);  // far off: reprojects badly everywhere
  EXPECT_EQ(filter_points(rec, idx, 4.0), 1);
  EXPECT_EQ(rec.has_point[3], 0);
  EXPECT_TRUE(triangulate_track(rec, idx, 3, p));  // and comes back from its observations

  p.init_min_inliers = 1000;  // more than any pair has
  EXPECT_THROW(reconstruct_incremental(in.pairs, in.features, idx, in.K, p, {}, nullptr),
               sfm::GeometryError);
}

TEST(Incremental, OneWrongObservationIsRejectedNotThePoint) {
  auto in = make_input(config(0.0));
  // Track 5 is seen by all 8 cameras; move its keypoint in image 3 far away (a wrong match).
  in.features[3].xy[std::size_t{2} * 5] += 60.0F;
  const TrackIndex idx(in.tracks, in.features);
  IncrementalParams p;
  Reconstruction rec = reconstruct_incremental(in.pairs, in.features, idx, in.K, p, {}, nullptr);
  ASSERT_EQ(rec.registered_count(), 8);
  ASSERT_EQ(rec.has_point[5], 1);  // the point survives (before: the whole track was rejected)
  int outliers = 0;
  for (const TrackObs* o = idx.begin(5); o != idx.end(5); ++o) {
    const bool used = observation_used(rec, idx, o);
    EXPECT_EQ(used, o->image != 3) << o->image;
    outliers += used ? 0 : 1;
  }
  EXPECT_EQ(outliers, 1);
  // The point agrees with every other view (the reconstruction has its own frame, so compare
  // reprojections, not coordinates).
  for (const TrackObs* o = idx.begin(5); o != idx.end(5); ++o) {
    if (o->image == 3) continue;
    const Camera& c = rec.cameras[static_cast<std::size_t>(o->image)];
    EXPECT_LT(reprojection_error(c.K, c.pose, rec.points[5], o->uv), 1e-3) << o->image;
  }
  // The rejected observation is not in the BA / metrics observations.
  for (const PointObs& ob : point_observations(rec, idx)) {
    EXPECT_FALSE(ob.track == 5 && ob.camera == 3);
  }
}

TEST(Incremental, FilterRemovesBadObservationsAndUnsupportedPoints) {
  const auto in = make_input(config(0.0));
  const TrackIndex idx(in.tracks, in.features);
  Reconstruction rec =
      reconstruct_incremental(in.pairs, in.features, idx, in.K, IncrementalParams{}, {}, nullptr);
  // Camera 2 moved: its observations of every point now reproject badly; others still agree.
  rec.cameras[2].pose.t += Eigen::Vector3d(5, 0, 0);
  const int removed = filter_points(rec, idx, 4.0);
  EXPECT_EQ(removed, 0);  // every point keeps >= 2 good views
  for (const TrackObs* o = idx.begin(0); o != idx.end(0); ++o) {
    EXPECT_EQ(observation_used(rec, idx, o), o->image != 2) << o->image;
  }
  // Move all cameras but one: points lose support and are removed.
  for (int c = 1; c < 8; ++c) {
    rec.cameras[static_cast<std::size_t>(c)].pose.t += Eigen::Vector3d(0, 7, 0);
  }
  EXPECT_EQ(filter_points(rec, idx, 4.0), 200);
  EXPECT_EQ(rec.point_count(), 0);
}
