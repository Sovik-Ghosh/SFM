// Runs with CUDA_VISIBLE_DEVICES set to empty (see CMakeLists.txt): CUDA sees no device.
#include <gtest/gtest.h>

#include <sstream>

#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#include "run_util.hpp"

using namespace sfm::pipeline;  // NOLINT(google-build-using-namespace)

TEST(BackendNoGpu, AutoFallsBackToTheCpuResultExactly) {
  const Backend b = detect_backend();
  EXPECT_FALSE(b.gpu);
  EXPECT_TRUE(b.device.empty());

  std::ostringstream text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &text, false);
  // 12 images: the CPU SIFT takes about 1 s per image.
  auto cpu = test::bunny_config("bunny.json", "sfm_backend_nogpu_cpu");
  cpu.features.method = "sift_cpu";
  cpu.dataset.last = 11;
  const auto a = run_pipeline(cpu, {}, log);
  auto aut = test::bunny_config("bunny_gpu.json", "sfm_backend_nogpu_auto");
  aut.run.name = cpu.run.name;  // metrics.json does not contain the name, but keep runs alike
  aut.dataset.last = 11;
  aut.features.method = "auto";
  aut.matching.method = "auto";
  aut.ba.method = "auto";
  const auto b2 = run_pipeline(aut, {}, log);
  EXPECT_EQ(test::read_file(b2.run_dir / "metrics.json"),
            test::read_file(a.run_dir / "metrics.json"));
  const auto used = sfm::config::parse_config(test::read_file(b2.run_dir / "config.json"));
  EXPECT_EQ(used.features.method, "sift_cpu");
  EXPECT_EQ(used.matching.method, "bf_cpu");
  EXPECT_EQ(used.ba.method, "lm_schur_cpu");
  std::filesystem::remove_all(cpu.run.output_dir);
  std::filesystem::remove_all(aut.run.output_dir);
}
