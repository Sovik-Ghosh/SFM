// Local bundle adjustment (ba::bundle_adjust with free_images) and the local / global schedule of
// the incremental loop.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/ba/adjust.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/recon/incremental.hpp"

#include "../recon/scene.hpp"

using namespace sfm::recon;  // NOLINT(google-build-using-namespace)

namespace {

sfm::eval::SyntheticConfig scene(double noise) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 8;
  c.points = 200;
  c.box_half = 20.0;
  c.noise_px = noise;
  return c;
}

double reproj_rms(const Reconstruction& rec, const TrackIndex& idx) {
  double s = 0.0;
  const auto e = reprojection_errors(rec, idx);
  for (const double v : e) s += v * v;
  return std::sqrt(s / static_cast<double>(e.size()));
}

}  // namespace

TEST(LocalBa, MovesOnlyTheFreeCamerasAndFitsThem) {
  const auto in = test::make_input(scene(0.5));
  const TrackIndex idx(in.tracks, in.features);
  Reconstruction rec =
      reconstruct_incremental(in.pairs, in.features, idx, in.K, IncrementalParams{}, {}, nullptr);
  ASSERT_EQ(rec.registered_count(), 8);
  sfm::ba::bundle_adjust(rec, idx, sfm::ba::Options{});  // a fitted global solution first
  const double fitted = reproj_rms(rec, idx);
  const Reconstruction fitted_rec = rec;

  // Disturb the last n registered cameras, then adjust only them. n = 7 leaves one fixed camera:
  // the scale is then held by a translation component of the first free camera.
  for (const std::ptrdiff_t n : {3, 7}) {
    SCOPED_TRACE(n);
    rec = fitted_rec;
    const std::vector<int> free(rec.order.end() - n, rec.order.end());
    sfm::eval::Rng rng(7);
    for (const int i : free) {
      Pose& pose = rec.cameras[static_cast<std::size_t>(i)].pose;
      const Eigen::Vector3d w(rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01),
                              rng.uniform(-0.01, 0.01));
      pose.R = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix() * pose.R;
      pose.t += Eigen::Vector3d(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0);
    }
    const Reconstruction before = rec;
    ASSERT_GT(reproj_rms(rec, idx), 5.0 * fitted);
    const auto report = sfm::ba::bundle_adjust(rec, idx, sfm::ba::Options{}, &sfm::ba::solve, free);
    EXPECT_LT(report.final_cost, report.initial_cost);
    for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
      const bool is_free = std::find(free.begin(), free.end(), static_cast<int>(i)) != free.end();
      if (is_free) continue;
      EXPECT_EQ(rec.cameras[i].pose.R, before.cameras[i].pose.R) << i;  // bit for bit
      EXPECT_EQ(rec.cameras[i].pose.t, before.cameras[i].pose.t) << i;
    }
    EXPECT_LT(reproj_rms(rec, idx), 1.05 * fitted);  // back to the fitted solution
  }
}

TEST(LocalBa, ScheduleAlternatesLocalAndGlobalAndEndsGlobal) {
  const auto in = test::make_input(scene(0.3));
  const TrackIndex idx(in.tracks, in.features);
  const auto run = [&](double growth) {
    IncrementalParams p;
    p.ba_every_n_cameras = 1;
    p.ba_local_cameras = 2;
    p.ba_global_growth = growth;
    std::vector<std::pair<int, int>> calls;  // (registered cameras, free images; 0 = global)
    const BundleAdjuster ba = [&](Reconstruction& r, const std::vector<int>& free_images) {
      calls.emplace_back(r.registered_count(), static_cast<int>(free_images.size()));
      if (!free_images.empty()) {  // the most recently registered cameras
        EXPECT_EQ(free_images, std::vector<int>(r.order.end() - 2, r.order.end()));
      }
    };
    IncrementalReport rep;
    reconstruct_incremental(in.pairs, in.features, idx, in.K, p, ba, nullptr, &rep);
    EXPECT_EQ(rep.ba_runs, static_cast<int>(calls.size()));
    return calls;
  };
  using V = std::vector<std::pair<int, int>>;
  // growth 0.5: global at 2, 3 (>= 1.5 * 2), 5 (>= 4.5), 8 (>= 7.5); local in between.
  EXPECT_EQ(run(0.5), (V{{2, 0}, {3, 0}, {4, 2}, {5, 0}, {6, 2}, {7, 2}, {8, 0}}));
  // growth 10: local after the initial pair until the end, then one final global BA.
  EXPECT_EQ(run(10.0), (V{{2, 0}, {3, 2}, {4, 2}, {5, 2}, {6, 2}, {7, 2}, {8, 2}, {8, 0}}));
}
