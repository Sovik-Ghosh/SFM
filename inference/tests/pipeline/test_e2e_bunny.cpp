// Bunny end-to-end regression (label "slow": GPU runner and local runs only). Gate of v0.2.0
// (P2 design): >= 31/36 cameras registered, median reprojection <= 1 px, deterministic, and no
// regression against the stored baseline (inference/tests/golden/p2/baseline_metrics.json,
// written by sfm_run with configs/bunny.json and reviewed like a golden file). Pose accuracy is
// compared with the baseline, not gated absolutely: bunny's narrow view limits it.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#include "test_env.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

std::string read_file(const fs::path& f) {
  std::ifstream in(f);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

TEST(E2eBunny, MeetsTheGateAndTheBaselineDeterministically) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / "bunny.json", src);
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_e2e_bunny";
  cfg.output.dump_stages = false;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);

  const auto first = sfm::pipeline::run_pipeline(cfg, {}, log);
  ASSERT_TRUE(first.metrics.has_value());
  const std::string metrics1 = read_file(first.run_dir / "metrics.json");
  const json m = json::parse(metrics1);
  const json base = json::parse(read_file(src / "inference/tests/golden/p2/baseline_metrics.json"));

  // Absolute gate.
  EXPECT_GE(m["registered"].get<int>(), 31);
  EXPECT_LE(m["reprojection_px"]["median"].get<double>(), 1.0);
  // Regression against the baseline.
  EXPECT_GE(m["registered"].get<int>(), base["registered"].get<int>());
  EXPECT_GE(m["points"].get<double>(), 0.95 * base["points"].get<double>());
  EXPECT_LE(m["reprojection_px"]["median"].get<double>(),
            base["reprojection_px"]["median"].get<double>() + 0.05);
  EXPECT_LE(m["rotation_error_deg"]["median"].get<double>(),
            base["rotation_error_deg"]["median"].get<double>() + 0.5);
  EXPECT_LE(m["centre_error_pct"]["median"].get<double>(),
            base["centre_error_pct"]["median"].get<double>() + 1.0);

  // Budget: < 30 s for the whole run (release build; meaningless under sanitizers / debug).
  if (!sfm::test::under_sanitizer()) {
    RecordProperty("total_seconds", std::to_string(first.timings.total));
  }

  // Deterministic: a second run gives a byte-identical metrics.json.
  const auto second = sfm::pipeline::run_pipeline(cfg, {}, log);
  EXPECT_EQ(read_file(second.run_dir / "metrics.json"), metrics1);
  fs::remove_all(cfg.run.output_dir);
}
