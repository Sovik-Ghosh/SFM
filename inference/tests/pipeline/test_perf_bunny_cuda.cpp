// P9c performance gates that run on the GPU CI job (bunny is in the repository). Time = the
// pipeline's own wall time (load to export), best of 3 runs after a warm-up run (CUDA context,
// first allocations), so a busy runner does not fail the gate on one slow run. Measured on the
// RTX 5060 Ti + Ryzen 7 9700X (2026-11-22, best of 3 warm runs): bunny_gpu 0.61 s,
// bunny_sift_tuned2 0.77 s, peak RSS 0.39 / 0.40 GB. Budgets: about 2.5x the time (CI runs in a
// container on the same runner), 2.5x the memory. Skipped under compute-sanitizer.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#include "run_util.hpp"
#include "test_env.hpp"

namespace {

void perf_gate(const std::string& config, int registered, double max_seconds, double max_rss_gb) {
  if (sfm::test::under_sanitizer()) GTEST_SKIP() << "timing is meaningless under the sanitizer";
  auto cfg = sfm::pipeline::test::bunny_config(config, "sfm_perf_" + config);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  double best = std::numeric_limits<double>::infinity();
  for (int run = 0; run < 4; ++run) {
    const auto r = sfm::pipeline::run_pipeline(cfg, {}, log);
    ASSERT_TRUE(r.metrics.has_value());
    const auto m = r.metrics.value_or(sfm::pipeline::Metrics{});
    EXPECT_EQ(static_cast<int>(m.registered_ids.size()), registered);
    if (run > 0) best = std::min(best, r.timings.total);  // run 0 warms up
  }
  const double rss = sfm::pipeline::test::peak_rss_gb();
  ::testing::Test::RecordProperty("best_s", std::to_string(best));
  ::testing::Test::RecordProperty("peak_rss_gb", std::to_string(rss));
  std::printf("%s: best %.3f s, peak RSS %.2f GB\n", config.c_str(), best, rss);
  EXPECT_LE(best, max_seconds) << "time budget";
  EXPECT_LE(rss, max_rss_gb) << "memory budget";
  std::filesystem::remove_all(cfg.run.output_dir);
}

}  // namespace

// ORB, matching and BA on the GPU.
TEST(PerfBunnyCuda, AllGpuStagesWithinBudget) { perf_gate("bunny_gpu.json", 35, 1.6, 1.0); }

// The most accurate bunny config (SIFT on the GPU).
TEST(PerfBunnyCuda, TunedSift2WithinBudget) { perf_gate("bunny_sift_tuned2.json", 36, 1.9, 1.0); }
