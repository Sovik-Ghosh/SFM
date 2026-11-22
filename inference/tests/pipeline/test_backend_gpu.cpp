#include <gtest/gtest.h>

#include <sstream>

#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#include "run_util.hpp"

using namespace sfm::pipeline;  // NOLINT(google-build-using-namespace)

TEST(Backend, GpuIsDetectedAndAutoUsesIt) {
  const Backend b = detect_backend();
  ASSERT_TRUE(b.gpu);  // this test runs on GPU machines only (label cuda)
  EXPECT_FALSE(b.device.empty());

  auto cfg = test::bunny_config("bunny.json", "sfm_backend_gpu");
  cfg.features.method = "auto";
  cfg.matching.method = "auto";
  cfg.ba.method = "auto";
  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kInfo, &text, false);
  const auto r = run_pipeline(cfg, {}, log);
  // The effective config records what "auto" chose.
  const auto used = sfm::config::parse_config(test::read_file(r.run_dir / "config.json"));
  EXPECT_EQ(used.features.method, "sift_cuda");
  EXPECT_EQ(used.matching.method, "bf_cuda");
  EXPECT_EQ(used.ba.method, "lm_schur_cuda");
  EXPECT_NE(text.str().find("event=run.backend gpu=true"), std::string::npos) << text.str();
  std::filesystem::remove_all(cfg.run.output_dir);
}
