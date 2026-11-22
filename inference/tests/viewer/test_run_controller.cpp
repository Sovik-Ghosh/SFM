// RunController with a fake pipeline (no GPU, no files): ordering, cancellation, refusal of a
// second run, failures, and the backend mapping. Also built in the tsan preset.
#include <gtest/gtest.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/viewer/run_controller.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)
using sfm::pipeline::ProgressSink;
using sfm::pipeline::SceneSnapshot;
using sfm::pipeline::Stage;
using Clock = std::chrono::steady_clock;

namespace {

// Snapshot k: k + 2 cameras, 100 * (k + 1) points whose coordinates all equal k (a torn copy
// would mix values).
SceneSnapshot fake_snapshot(int k) {
  SceneSnapshot s;
  s.sequence = k;
  for (int i = 0; i < k + 2; ++i) s.order.push_back(i);
  s.points.assign(std::size_t{100} * static_cast<std::size_t>(k + 1), Eigen::Vector3d::Constant(k));
  return s;
}

// A pipeline that registers `n` cameras, one every `step`, checking for cancellation before each.
RunController::Runner fake_pipeline(int n, std::chrono::milliseconds step) {
  return [n, step](ProgressSink& sink) {
    sink.stage_begin(Stage::kReconstruction);
    for (int k = 0; k < n; ++k) {
      if (sink.cancelled()) throw sfm::Cancelled("cancelled");
      std::this_thread::sleep_for(step);
      if (k % 5 == 4) {
        sink.ba_done(fake_snapshot(k), sfm::ba::Report{});
      } else {
        sink.camera_registered(fake_snapshot(k));
      }
    }
    sink.stage_end(Stage::kReconstruction, 0.5);
    sink.finished(sfm::pipeline::Metrics{}, sfm::pipeline::Timings{});
  };
}

// Polls until the run ends (or 5 s pass), collecting everything.
RunUpdate drain(RunController& c) {
  RunUpdate all;
  const auto deadline = Clock::now() + std::chrono::seconds(5);
  while (Clock::now() < deadline) {
    RunUpdate u = c.poll();
    for (auto& s : u.snapshots) all.snapshots.push_back(std::move(s));
    for (auto& e : u.events) all.events.push_back(std::move(e));
    if (u.metrics) all.metrics = u.metrics;
    all.state = u.state;
    if (u.state != RunState::kRunning) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return all;
}

}  // namespace

TEST(RunController, SnapshotsArriveInOrderAndWhole) {
  RunController c;
  ASSERT_TRUE(c.start(fake_pipeline(20, std::chrono::milliseconds(1))));
  const RunUpdate u = drain(c);
  EXPECT_EQ(u.state, RunState::kFinished);
  ASSERT_EQ(u.snapshots.size(), 20U);
  for (std::size_t k = 0; k < u.snapshots.size(); ++k) {
    const SceneSnapshot& s = u.snapshots[k];
    EXPECT_EQ(s.sequence, static_cast<int>(k));
    ASSERT_EQ(s.points.size(), 100 * (k + 1));
    for (const auto& p : s.points) ASSERT_EQ(p, Eigen::Vector3d::Constant(static_cast<double>(k)));
  }
  // Events: stage begin, 16 cameras + 4 BAs (each with its snapshot index), stage end, finished.
  ASSERT_EQ(u.events.size(), 23U);
  EXPECT_EQ(u.events.front().kind, EventKind::kStageBegin);
  EXPECT_EQ(u.events[u.events.size() - 2].kind, EventKind::kStageEnd);
  EXPECT_DOUBLE_EQ(u.events[u.events.size() - 2].seconds, 0.5);
  EXPECT_EQ(u.events.back().kind, EventKind::kFinished);
  int ba = 0;
  for (std::size_t i = 1; i + 2 < u.events.size(); ++i) {
    const RunEvent& e = u.events[i];
    EXPECT_EQ(e.snapshot, static_cast<int>(i - 1));
    EXPECT_EQ(e.cameras, static_cast<int>(i + 1));
    ba += e.kind == EventKind::kBundleAdjustment ? 1 : 0;
    if (e.kind == EventKind::kCamera) {
      EXPECT_EQ(e.newest_image, static_cast<int>(i));
    }
  }
  EXPECT_EQ(ba, 4);
  EXPECT_TRUE(u.metrics.has_value());
  EXPECT_GE(c.elapsed_seconds(), 0.02);
}

TEST(RunController, CancelStopsAtTheNextCheck) {
  RunController c;
  ASSERT_TRUE(c.start(fake_pipeline(1000, std::chrono::milliseconds(20))));
  // Wait for the first snapshot, then cancel.
  const auto deadline = Clock::now() + std::chrono::seconds(5);
  std::size_t seen = 0;
  while (seen == 0 && Clock::now() < deadline) {
    seen += c.poll().snapshots.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(seen, 0U);
  const auto t0 = Clock::now();
  c.cancel();
  const RunUpdate u = drain(c);
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  EXPECT_EQ(u.state, RunState::kCancelled);
  ASSERT_FALSE(u.events.empty());
  EXPECT_EQ(u.events.back().kind, EventKind::kCancelled);
  EXPECT_LT(ms, 100.0);  // one step (20 ms) plus scheduling
  EXPECT_FALSE(u.metrics.has_value());
}

TEST(RunController, SecondStartIsRefusedWhileRunningAndAllowedAfter) {
  RunController c;
  ASSERT_TRUE(c.start(fake_pipeline(1000, std::chrono::milliseconds(5))));
  EXPECT_TRUE(c.active());
  EXPECT_FALSE(c.start(fake_pipeline(1, std::chrono::milliseconds(0))));
  c.cancel();
  EXPECT_EQ(drain(c).state, RunState::kCancelled);
  ASSERT_TRUE(c.start(fake_pipeline(3, std::chrono::milliseconds(0))));
  const RunUpdate u = drain(c);
  EXPECT_EQ(u.state, RunState::kFinished);
  ASSERT_EQ(u.snapshots.size(), 3U);
  EXPECT_EQ(u.snapshots.front().sequence, 0);  // a new run starts a new sequence
}

TEST(RunController, FailureIsReportedWithItsMessage) {
  RunController c;
  ASSERT_TRUE(c.start([](ProgressSink& sink) {
    sink.stage_begin(Stage::kFeatures);
    throw sfm::IoError("images/0000.ppm: not found");
  }));
  const RunUpdate u = drain(c);
  EXPECT_EQ(u.state, RunState::kFailed);
  ASSERT_EQ(u.events.size(), 2U);
  EXPECT_EQ(u.events.back().kind, EventKind::kFailed);
  EXPECT_NE(u.events.back().message.find("not found"), std::string::npos);
}

TEST(RunController, DestructorCancelsAndJoins) {
  const auto t0 = Clock::now();
  {
    RunController c;
    ASSERT_TRUE(c.start(fake_pipeline(100000, std::chrono::milliseconds(10))));
  }
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  EXPECT_LT(ms, 200.0);
}

TEST(RunController, PollBeforeAnyRunIsIdle) {
  RunController c;
  const RunUpdate u = c.poll();
  EXPECT_EQ(u.state, RunState::kIdle);
  EXPECT_TRUE(u.snapshots.empty());
  EXPECT_FALSE(c.active());
}

TEST(WithBackend, MapsMethodNames) {
  sfm::config::Config cfg;
  cfg.features.method = "sift_cuda";
  cfg.matching.method = "auto";
  cfg.ba.method = "lm_schur_cpu";
  const auto cpu = with_backend(cfg, BackendChoice::kCpu, true);
  EXPECT_EQ(cpu.features.method, "sift_cpu");
  EXPECT_EQ(cpu.matching.method, "bf_cpu");
  EXPECT_EQ(cpu.ba.method, "lm_schur_cpu");
  const auto gpu = with_backend(cfg, BackendChoice::kGpu, true);
  EXPECT_EQ(gpu.features.method, "sift_cuda");
  EXPECT_EQ(gpu.matching.method, "bf_cuda");
  EXPECT_EQ(gpu.ba.method, "lm_schur_cuda");
  const auto autoc = with_backend(cfg, BackendChoice::kAuto, false);
  EXPECT_EQ(autoc.features.method, "sift_cuda");  // explicit names are kept by "auto"
  EXPECT_EQ(autoc.matching.method, "bf_cpu");
  EXPECT_EQ(autoc.ba.method, "lm_schur_cpu");
}
