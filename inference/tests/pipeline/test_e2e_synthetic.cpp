// End to end on synthetic scenes (exact ground truth): reconstruction + BA from perfect tracks.
// This gates correctness; the bunny run gates registration and regression (see the P2 design).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>

#include "sfm/ba/adjust.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/recon/incremental.hpp"

#include "scene.hpp"

namespace {

struct Result {
  int registered = 0;
  sfm::eval::PoseErrors errors;
  double reproj_median = 0.0;
};

Result run(double box_half, double noise, std::uint64_t seed) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 12;
  c.points = 300;
  c.box_half = box_half;
  c.noise_px = noise;
  c.seed = seed;
  const auto in = sfm::recon::test::make_input(c);
  const sfm::recon::TrackIndex idx(in.tracks, in.features);
  sfm::recon::IncrementalParams p;
  p.seed = seed;
  const sfm::recon::BundleAdjuster ba = [&](sfm::recon::Reconstruction& r,
                                            const std::vector<int>& free_images) {
    sfm::ba::bundle_adjust(r, idx, sfm::ba::Options{}, &sfm::ba::solve, free_images);
  };
  const auto rec =
      sfm::recon::reconstruct_incremental(in.pairs, in.features, idx, in.K, p, ba, nullptr);
  Result res;
  res.registered = rec.registered_count();
  std::vector<sfm::eval::Pose> est;
  std::vector<sfm::eval::Pose> tru;
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    if (!rec.cameras[i].registered) continue;
    est.push_back(rec.cameras[i].pose);
    tru.push_back(in.scene.poses[i]);
  }
  res.errors = sfm::eval::pose_error(est, tru);
  auto e = sfm::recon::reprojection_errors(rec, idx);
  std::nth_element(e.begin(), e.begin() + static_cast<std::ptrdiff_t>(e.size() / 2), e.end());
  res.reproj_median = e[e.size() / 2];
  return res;
}

}  // namespace

TEST(E2eSynthetic, NoiseFreeIsExact) {
  const Result r = run(20.0, 0.0, 1);
  EXPECT_EQ(r.registered, 12);
  // Keypoints are stored as float (as in the real pipeline): ~3e-5 px of rounding.
  EXPECT_LT(r.errors.max_rotation_deg(), 1e-5);
  EXPECT_LT(r.errors.max_centre_pct(), 1e-5);
  EXPECT_LT(r.reproj_median, 1e-4);
}

TEST(E2eSynthetic, WideFieldWithNoiseIsAccurate) {
  // Two-view relative pose alone gives a 0.2 deg median error here (P2 conditioning study);
  // 12 views + BA must do better. Bounds: measured 2026-11-19 over 3 seeds (max rotation
  // 0.0175 / 0.0162 / 0.0162 deg, max centre 0.033 / 0.037 / 0.034 %), times ~3.
  for (std::uint64_t seed : {1, 2, 3}) {
    const Result r = run(20.0, 0.5, seed);
    std::printf(
        "seed %llu: max rotation %.4f deg, max centre %.4f %%, reprojection median %.3f px\n",
        static_cast<unsigned long long>(seed), r.errors.max_rotation_deg(),
        r.errors.max_centre_pct(), r.reproj_median);
    EXPECT_EQ(r.registered, 12);
    EXPECT_LT(r.errors.max_rotation_deg(), 0.06);
    EXPECT_LT(r.errors.max_centre_pct(), 0.12);
    EXPECT_LT(r.reproj_median, 0.7);  // ~ noise sigma * sqrt(2 ln 2) = 0.59 for 2-D Gaussian
  }
}

TEST(E2eSynthetic, BunnyLikeNarrowFieldStillRegistersEverything) {
  const Result r = run(5.0, 0.5, 1);
  EXPECT_EQ(r.registered, 12);
  EXPECT_LT(r.errors.max_rotation_deg(), 1.0);
}
