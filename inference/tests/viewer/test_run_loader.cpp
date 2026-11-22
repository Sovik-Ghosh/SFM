// A finished run loaded from disk gives the same snapshot as the live run's last one.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>

#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/viewer/history.hpp"
#include "sfm/viewer/run_controller.hpp"
#include "sfm/viewer/run_loader.hpp"

namespace fs = std::filesystem;
using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

namespace {

struct Keep : sfm::pipeline::ProgressSink {
  SnapshotHistory history;
  void camera_registered(const sfm::pipeline::SceneSnapshot& s) override { history.push(s); }
  void ba_done(const sfm::pipeline::SceneSnapshot& s, const sfm::ba::Report&) override {
    history.push(s);
  }
};

}  // namespace

TEST(RunLoader, LoadedRunEqualsTheLiveFinalSnapshot) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_gpu.json", src);
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_viewer_load";
  cfg.output.ply = false;
  cfg.output.colmap = false;
  cfg.output.dump_stages = true;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &text, false);
  Keep keep;
  const auto run =
      sfm::pipeline::run_pipeline(cfg, {}, log, sfm::pipeline::default_registry(), &keep);
  ASSERT_FALSE(keep.history.empty());
  for (std::size_t k = 0; k < keep.history.size(); ++k) {
    EXPECT_EQ(keep.history.at(k).sequence, static_cast<int>(k));
  }

  const LoadedRun r = load_run(run.run_dir);
  const auto& live = keep.history.latest();
  const auto& disk = r.snapshot;
  EXPECT_EQ(disk.ids, live.ids);
  EXPECT_EQ(disk.registered, live.registered);
  EXPECT_EQ(disk.order, live.order);
  EXPECT_EQ(disk.R, live.R);
  EXPECT_EQ(disk.t, live.t);
  EXPECT_EQ(disk.points, live.points);
  EXPECT_EQ(disk.point_gray, live.point_gray);
  EXPECT_EQ(disk.point_error_px, live.point_error_px);
  EXPECT_EQ(disk.point_track_length, live.point_track_length);
  EXPECT_EQ(r.truth.size(), 36u);
  EXPECT_NE(r.metrics_json.find("\"registered\""), std::string::npos);
  EXPECT_EQ(r.config.features.method, "orb_cuda");
  // Replay from disk: cameras in registration order, consecutive sequences, growing point sets,
  // the last step equal to the final state.
  ASSERT_GE(r.replay.size(), 2u);
  for (std::size_t k = 0; k < r.replay.size(); ++k) {
    const auto& s = r.replay[k];
    EXPECT_EQ(s.sequence, static_cast<int>(k));
    ASSERT_LE(s.order.size(), disk.order.size());
    EXPECT_TRUE(std::equal(s.order.begin(), s.order.end(), disk.order.begin()));
    if (k > 0) {
      EXPECT_GT(s.order.size(), r.replay[k - 1].order.size());
      EXPECT_GE(s.points.size(), r.replay[k - 1].points.size());
    }
  }
  EXPECT_EQ(r.replay.front().order.size(), 2u);
  EXPECT_EQ(r.replay.back().order, disk.order);
  EXPECT_EQ(r.replay.back().points, disk.points);
  EXPECT_EQ(r.replay.back().registered, disk.registered);

  EXPECT_THROW(load_run(cfg.run.output_dir / "missing"), sfm::IoError);
  fs::remove_all(run.run_dir / "stages" / "tracks");
  EXPECT_THROW(load_run(run.run_dir), sfm::IoError);
  fs::remove_all(cfg.run.output_dir);
}

TEST(SnapshotHistory, RejectsGapsAndClamps) {
  SnapshotHistory h;
  sfm::pipeline::SceneSnapshot s;
  s.sequence = 0;
  h.push(s);
  s.sequence = 2;
  EXPECT_THROW(h.push(s), std::invalid_argument);
  s.sequence = 1;
  h.push(s);
  EXPECT_EQ(h.size(), 2u);
  EXPECT_EQ(h.at(99).sequence, 1);
}

// The real pipeline through the RunController: the viewer's live run writes the same metrics as
// a direct run, and its events match the metrics (one per camera after the pair, one per BA).
TEST(RunController, LiveBunnyRunMatchesTheDirectRun) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_gpu.json", src);
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_viewer_live";
  cfg.output.ply = false;
  cfg.output.colmap = false;
  cfg.output.dump_stages = false;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &text, false);
  const auto direct = sfm::pipeline::run_pipeline(cfg, {}, log);
  const std::string want =
      sfm::pipeline::to_json(direct.metrics.value_or(sfm::pipeline::Metrics{}));

  RunController c;
  ASSERT_TRUE(c.start(pipeline_runner(cfg)));
  RunUpdate all;
  for (int i = 0; i < 6000; ++i) {
    RunUpdate u = c.poll();
    for (auto& s : u.snapshots) all.snapshots.push_back(std::move(s));
    for (auto& e : u.events) all.events.push_back(std::move(e));
    if (u.metrics) all.metrics = u.metrics;
    all.state = u.state;
    if (u.state != RunState::kRunning) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_EQ(all.state, RunState::kFinished);
  ASSERT_TRUE(all.metrics.has_value());
  const sfm::pipeline::Metrics m = all.metrics.value_or(sfm::pipeline::Metrics{});
  EXPECT_EQ(sfm::pipeline::to_json(m), want);
  int cameras = 0;
  int bas = 0;
  int stage_ends = 0;
  for (const RunEvent& e : all.events) {
    cameras += e.kind == EventKind::kCamera ? 1 : 0;
    bas += e.kind == EventKind::kBundleAdjustment ? 1 : 0;
    stage_ends += e.kind == EventKind::kStageEnd ? 1 : 0;
  }
  EXPECT_EQ(cameras, static_cast<int>(m.registered_ids.size()) - 1);
  EXPECT_EQ(bas, m.ba_runs);
  EXPECT_EQ(stage_ends, 5);
  EXPECT_EQ(static_cast<int>(all.snapshots.size()), cameras + bas);
  fs::remove_all(cfg.run.output_dir);
}

// names.txt of a rig import (tools/import_eth3d.py --order time): one group per camera folder, in
// folder order, labelled by the camera; one folder or a gap: no groups.
TEST(RunLoader, CameraGroupsFromNames) {
  const std::string names =
      "0000 images_rig_cam4_undistorted/1.png\n0001 images_rig_cam5_undistorted/1.png\n"
      "0002 images_rig_cam4_undistorted/2.png\n0003 images_rig_cam5_undistorted/2.png\n";
  const auto g = sfm::viewer::camera_groups(names, 0, 3);
  EXPECT_EQ(g.group, (std::vector<int>{0, 1, 0, 1}));
  EXPECT_EQ(g.labels, (std::vector<std::string>{"cam4", "cam5"}));
  EXPECT_EQ(sfm::viewer::camera_groups(names, 1, 2).group, (std::vector<int>{1, 0}));
  EXPECT_TRUE(sfm::viewer::camera_groups("0000 a/1.png\n0001 a/2.png\n", 0, 1).group.empty());
  EXPECT_TRUE(sfm::viewer::camera_groups(names, 0, 5).group.empty());  // 4, 5 missing
  EXPECT_TRUE(sfm::viewer::camera_groups("", 0, 3).group.empty());
  const auto plain = sfm::viewer::camera_groups("0000 left/1.png\n0001 right/1.png\n", 0, 1);
  EXPECT_EQ(plain.labels, (std::vector<std::string>{"left", "right"}));
}
