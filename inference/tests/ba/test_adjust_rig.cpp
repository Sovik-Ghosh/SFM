// Rigid rig (P9b): the incremental loop with reconstruction.rig and the rig bundle adjustment on a
// synthetic 2-sensor rig.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/ba/adjust.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/recon/incremental.hpp"

#include "../recon/scene.hpp"

using namespace sfm::recon;  // NOLINT(google-build-using-namespace)

namespace {

// Frames = the poses of a synthetic ring; 2 sensors: the reference and one 0.8 units to its side,
// turned by ~2 degrees. Image 2 f + s = sensor s of frame f.
struct RigScene {
  sfm::eval::SyntheticScene scene;  // one pose per image
  std::shared_ptr<sfm::data::Rig> rig;
};

RigScene make_rig_scene(double noise) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 7;
  c.points = 250;
  c.box_half = 20.0;
  c.noise_px = 0.0;
  const auto ring = sfm::eval::make_synthetic_scene(c);
  RigScene out;
  out.rig = std::make_shared<sfm::data::Rig>();
  out.rig->R = {Eigen::Matrix3d::Identity(), exp_so3(Eigen::Vector3d(0.01, 0.03, -0.01))};
  out.rig->t = {Eigen::Vector3d::Zero(), Eigen::Vector3d(-0.8, 0.05, 0.0)};
  out.rig->frames = c.cameras;
  out.scene.K = ring.K;
  out.scene.points = ring.points;
  sfm::eval::Rng rng(5);
  for (int f = 0; f < c.cameras; ++f) {
    for (int s = 0; s < 2; ++s) {
      out.rig->frame.push_back(f);
      out.rig->sensor.push_back(s);
      const auto i = static_cast<int>(out.scene.poses.size());
      out.scene.poses.push_back(
          rig_image_pose(*out.rig, i, ring.poses[static_cast<std::size_t>(f)]));
    }
  }
  for (std::size_t i = 0; i < out.scene.poses.size(); ++i) {
    const auto& P = out.scene.poses[i];
    for (Eigen::Index k = 0; k < ring.points.cols(); ++k) {
      const Eigen::Vector3d h = ring.K * (P.R * ring.points.col(k) + P.t);
      sfm::eval::Observation o;
      o.camera = static_cast<int>(i);
      o.point = static_cast<int>(k);
      o.uv = h.head<2>() / h.z() + noise * Eigen::Vector2d(rng.normal(), rng.normal());
      out.scene.observations.push_back(o);
    }
  }
  return out;
}

}  // namespace

TEST(RigPose, FrameAndImagePosesAreInverse) {
  sfm::data::Rig rig;
  rig.R = {Eigen::Matrix3d::Identity(), exp_so3(Eigen::Vector3d(0.2, -0.1, 0.3))};
  rig.t = {Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 2, 3)};
  rig.frame = {0, 0};
  rig.sensor = {0, 1};
  const Pose frame{exp_so3(Eigen::Vector3d(-0.4, 0.5, 0.1)), Eigen::Vector3d(4, -1, 2)};
  const Pose img = rig_image_pose(rig, 1, frame);
  const Pose back = rig_frame_pose(rig, 1, img);
  EXPECT_LT((back.R - frame.R).norm(), 1e-12);
  EXPECT_LT((back.t - frame.t).norm(), 1e-12);
  EXPECT_LT((rig_image_pose(rig, 0, frame).R - frame.R).norm(), 0.0 + 1e-15);  // reference sensor
}

TEST(IncrementalRig, RegistersWholeFramesAtMetricScaleAndKeepsTheRig) {
  const RigScene rs = make_rig_scene(0.3);
  const auto in = sfm::recon::test::make_input(rs.scene);
  const TrackIndex idx(in.tracks, in.features);
  IncrementalParams p;
  p.rig = rs.rig;
  p.ba_every_n_cameras = 4;
  const BundleAdjuster ba = [&](Reconstruction& r, const std::vector<int>& free_images) {
    sfm::ba::Options o;
    sfm::ba::bundle_adjust(r, idx, o, &sfm::ba::solve, free_images);
  };
  IncrementalReport rep;
  Reconstruction rec =
      reconstruct_incremental(in.pairs, in.features, idx, in.K, p, ba, nullptr, &rep);
  ASSERT_EQ(rec.registered_count(), 14);
  EXPECT_GE(rep.rig_registered,
            7);  // one image per frame by PnP or the pair, its sibling by the rig
  EXPECT_GT(rep.rig_scale, 0.0);
  sfm::ba::Options o;
  sfm::ba::bundle_adjust(rec, idx, o, &sfm::ba::solve, {});
  for (std::size_t f = 0; f < 7; ++f) {
    // The two images of a frame keep the rig's offset exactly, and its metric baseline.
    const Pose& a = rec.cameras[2 * f].pose;
    const Pose& b = rec.cameras[2 * f + 1].pose;
    const Eigen::Matrix3d Rrel = b.R * a.R.transpose();
    EXPECT_LT((Rrel - rs.rig->R[1]).norm(), 1e-9) << f;
    EXPECT_LT((b.t - Rrel * a.t - rs.rig->t[1]).norm(), 1e-9) << f;
    EXPECT_NEAR((a.centre() - b.centre()).norm(), rs.rig->t[1].norm(), 1e-9) << f;
  }
  // Against the truth: small rotation errors and the true scale (similarity scale ~ 1).
  std::vector<sfm::eval::Pose> est;
  est.reserve(rec.cameras.size());
  for (const auto& cam : rec.cameras) est.push_back(cam.pose);
  const auto err = sfm::eval::pose_error(est, rs.scene.poses);
  std::vector<double> rot = err.rotation_deg;
  std::nth_element(rot.begin(), rot.begin() + static_cast<std::ptrdiff_t>(rot.size() / 2),
                   rot.end());
  EXPECT_LT(rot[rot.size() / 2], 0.1);
  EXPECT_NEAR(err.alignment.s, 1.0, 0.01);
}
