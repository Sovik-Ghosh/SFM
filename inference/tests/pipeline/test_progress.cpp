// Progress events and cancellation on bunny (GPU stages: fast).
#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#include "run_util.hpp"

using namespace sfm::pipeline;  // NOLINT(google-build-using-namespace)

namespace {

struct Recorder : ProgressSink {
  std::vector<std::string> events;
  std::vector<SceneSnapshot> snaps;
  int finished_calls = 0;
  int cancel_after_snaps = -1;  // cancel once this many snapshots arrived (-1: never)

  void stage_begin(Stage s) override { events.push_back("begin " + to_string(s)); }
  void stage_end(Stage s, double seconds) override {
    events.push_back("end " + to_string(s));
    EXPECT_GE(seconds, 0.0);
  }
  void camera_registered(const SceneSnapshot& s) override {
    events.push_back("camera");
    snaps.push_back(s);
  }
  void ba_done(const SceneSnapshot& s, const sfm::ba::Report& r) override {
    events.push_back("ba");
    snaps.push_back(s);
    EXPECT_LE(r.final_cost, r.initial_cost);
  }
  void finished(const Metrics&, const Timings&) override { ++finished_calls; }
  bool cancelled() const override {
    return cancel_after_snaps >= 0 && static_cast<int>(snaps.size()) >= cancel_after_snaps;
  }
};

int count(const std::vector<std::string>& v, const std::string& x) {
  int n = 0;
  for (const auto& e : v) n += e == x ? 1 : 0;
  return n;
}

}  // namespace

TEST(Progress, EventsAreOrderedConsistentAndPassive) {
  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &text, false);
  auto plain = test::bunny_config("bunny_gpu.json", "sfm_progress_plain");
  const auto a = run_pipeline(plain, {}, log);
  const std::string metrics_plain = test::read_file(a.run_dir / "metrics.json");

  auto cfg = test::bunny_config("bunny_gpu.json", "sfm_progress_sink");
  Recorder rec;
  const auto b = run_pipeline(cfg, {}, log, default_registry(), &rec);
  // A sink never changes the result.
  EXPECT_EQ(test::read_file(b.run_dir / "metrics.json"), metrics_plain);

  // Stage events: five begin / end pairs in pipeline order, reconstruction events inside.
  const std::vector<std::string> stages = {"features", "matching", "tracks", "reconstruction",
                                           "export"};
  std::vector<std::string> stage_events;
  for (const auto& e : rec.events) {
    if (e.rfind("begin", 0) == 0 || e.rfind("end", 0) == 0) stage_events.push_back(e);
  }
  std::vector<std::string> want;
  for (const auto& s : stages) {
    want.push_back("begin " + s);
    want.push_back("end " + s);
  }
  EXPECT_EQ(stage_events, want);

  ASSERT_TRUE(b.metrics.has_value());
  const Metrics m = b.metrics.value_or(Metrics{});
  const auto registered = static_cast<int>(m.registered_ids.size());
  // One event for the initial pair (two cameras), then one per registered camera.
  EXPECT_EQ(count(rec.events, "camera"), registered - 1);
  EXPECT_EQ(count(rec.events, "ba"), m.ba_runs);
  EXPECT_EQ(rec.finished_calls, 1);

  ASSERT_FALSE(rec.snaps.empty());
  for (std::size_t k = 0; k < rec.snaps.size(); ++k) {
    const SceneSnapshot& s = rec.snaps[k];
    EXPECT_EQ(s.sequence, static_cast<int>(k));
    EXPECT_EQ(s.ids.size(), 36u);
    EXPECT_EQ(s.points.size(), s.point_gray.size());
    EXPECT_EQ(s.points.size(), s.point_error_px.size());
    int reg = 0;
    for (const auto r : s.registered) reg += r;
    EXPECT_EQ(reg, static_cast<int>(s.order.size()));
  }
  EXPECT_EQ(rec.snaps.front().order.size(), 2u);  // initial pair
  const SceneSnapshot& last = rec.snaps.back();
  EXPECT_EQ(static_cast<int>(last.order.size()), registered);
  EXPECT_EQ(static_cast<int>(last.points.size()), m.points);
  std::filesystem::remove_all(plain.run.output_dir);
  std::filesystem::remove_all(cfg.run.output_dir);
}

TEST(Progress, CancelStopsTheRunAndTheNextRunWorks) {
  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &text, false);
  auto cfg = test::bunny_config("bunny_gpu.json", "sfm_progress_cancel");
  Recorder rec;
  rec.cancel_after_snaps = 1;  // right after the initial pair
  EXPECT_THROW(run_pipeline(cfg, {}, log, default_registry(), &rec), sfm::Cancelled);
  EXPECT_EQ(rec.snaps.size(), 1u);  // stopped at the next check
  EXPECT_EQ(rec.finished_calls, 0);
  EXPECT_FALSE(std::filesystem::exists(cfg.run.output_dir / cfg.run.name / "metrics.json"));

  Recorder before;  // cancel before any work
  before.cancel_after_snaps = 0;
  EXPECT_THROW(run_pipeline(cfg, {}, log, default_registry(), &before), sfm::Cancelled);
  EXPECT_TRUE(before.events.empty());

  Recorder ok;
  const auto r = run_pipeline(cfg, {}, log, default_registry(), &ok);
  EXPECT_EQ(ok.finished_calls, 1);
  EXPECT_TRUE(r.metrics.has_value());
  std::filesystem::remove_all(cfg.run.output_dir);
}
