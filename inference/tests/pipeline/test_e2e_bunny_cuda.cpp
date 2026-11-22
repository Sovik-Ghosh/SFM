// Bunny end to end with the GPU stages.
// - sift_cuda (configs/bunny_sift.json): COLMAP's accuracy (P7 gate), deterministic;
//   configs/bunny_sift_tuned.json: more points, better than COLMAP; bunny_sift_tuned2.json:
//   tighter verification and final loss, the most accurate.
// - orb_cuda: metrics.json must equal the CPU run's (the detector is bit-identical).
// - orb_cuda + bf_cuda: the GPU RANSAC samples differently, so the result is not identical; it must
//   meet the gate, stay within the regression tolerances of the (CPU) baseline, and be
//   deterministic.
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/krt.hpp"
#include "sfm/pipeline/pipeline.hpp"

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& f) {
  std::ifstream in(f);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

TEST(E2eBunnyCuda, OrbCudaGivesTheCpuResult) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / "bunny.json", src);
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_e2e_bunny_cuda";
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);

  cfg.features.method = "orb_cpu";
  const auto cpu = sfm::pipeline::run_pipeline(cfg, {}, log);
  const std::string cpu_metrics = read_file(cpu.run_dir / "metrics.json");
  cfg.features.method = "orb_cuda";
  const auto gpu = sfm::pipeline::run_pipeline(cfg, {}, log);
  EXPECT_EQ(read_file(gpu.run_dir / "metrics.json"), cpu_metrics);
  RecordProperty("features_cpu_s", std::to_string(cpu.timings.features));
  RecordProperty("features_cuda_s", std::to_string(gpu.timings.features));
  fs::remove_all(cfg.run.output_dir);
}

namespace {

void gpu_run_meets_gate_and_baseline(const std::string& ba_method) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / "bunny.json", src);
  // One directory per BA method: ctest runs the two tests at the same time, and a shared
  // directory let one test delete the other's metrics.json (flaky "cannot write ...").
  cfg.run.output_dir = fs::temp_directory_path() / ("sfm_e2e_bunny_gpu_matching_" + ba_method);
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  cfg.features.method = "orb_cuda";
  cfg.matching.method = "bf_cuda";
  cfg.ba.method = ba_method;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  const auto first = sfm::pipeline::run_pipeline(cfg, {}, log);
  const std::string metrics = read_file(first.run_dir / "metrics.json");
  const auto m = nlohmann::json::parse(metrics);
  const auto base =
      nlohmann::json::parse(read_file(src / "inference/tests/golden/p2/baseline_metrics.json"));
  EXPECT_GE(m["registered"].get<int>(), 31);
  EXPECT_LE(m["reprojection_px"]["median"].get<double>(), 1.0);
  EXPECT_GE(m["registered"].get<int>(), base["registered"].get<int>());
  EXPECT_GE(m["points"].get<double>(), 0.95 * base["points"].get<double>());
  EXPECT_LE(m["reprojection_px"]["median"].get<double>(),
            base["reprojection_px"]["median"].get<double>() + 0.05);
  EXPECT_LE(m["rotation_error_deg"]["median"].get<double>(),
            base["rotation_error_deg"]["median"].get<double>() + 0.5);
  EXPECT_LE(m["centre_error_pct"]["median"].get<double>(),
            base["centre_error_pct"]["median"].get<double>() + 1.0);
  const auto second = sfm::pipeline::run_pipeline(cfg, {}, log);
  EXPECT_EQ(read_file(second.run_dir / "metrics.json"), metrics);
  ::testing::Test::RecordProperty("total_s", std::to_string(first.timings.total));
  fs::remove_all(cfg.run.output_dir);
}

}  // namespace

TEST(E2eBunnyCuda, GpuMatchingMeetsTheGateAndTheBaseline) {
  gpu_run_meets_gate_and_baseline("lm_schur_cpu");
}

TEST(E2eBunnyCuda, AllGpuStagesMeetTheGateAndTheBaseline) {
  gpu_run_meets_gate_and_baseline("lm_schur_cuda");
}

namespace {

struct SiftGate {
  const char* config;
  double rot_median;
  double rot_max;
  double centre_max;
  int min_points;
};

// Runs a SIFT config twice: gate on registration and accuracy, metrics byte-identical.
void sift_run_meets_gate(const SiftGate& g) {
  const fs::path src(SFM_SOURCE_DIR);
  auto cfg = sfm::config::load_config(src / "configs" / g.config, src);
  cfg.run.output_dir = fs::temp_directory_path() / (std::string("sfm_e2e_") + cfg.run.name);
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  const auto first = sfm::pipeline::run_pipeline(cfg, {}, log);
  const std::string metrics = read_file(first.run_dir / "metrics.json");
  const auto m = nlohmann::json::parse(metrics);
  EXPECT_EQ(m["registered"].get<int>(), 36);
  EXPECT_GE(m["points"].get<int>(), g.min_points);
  EXPECT_LE(m["rotation_error_deg"]["median"].get<double>(), g.rot_median);
  EXPECT_LE(m["rotation_error_deg"]["max"].get<double>(), g.rot_max);
  EXPECT_LE(m["centre_error_pct"]["max"].get<double>(), g.centre_max);
  const auto second = sfm::pipeline::run_pipeline(cfg, {}, log);
  EXPECT_EQ(read_file(second.run_dir / "metrics.json"), metrics);
  ::testing::Test::RecordProperty("total_s", std::to_string(first.timings.total));
  fs::remove_all(cfg.run.output_dir);
}

}  // namespace

TEST(E2eBunnyCuda, SiftReachesColmapAccuracyDeterministically) {
  // COLMAP (pycolmap 4.2.1, SIFT, masks, true intrinsics fixed): 36/36, rotation 0.28 / 0.64 deg,
  // centre 0.44 / 1.07 % (shared/findings, 2026-11-20). Measured here (P8): 36/36, 3736 points,
  // 0.258 / 0.575 deg.
  sift_run_meets_gate({"bunny_sift.json", 0.35, 0.7, 1.2, 3000});
}

TEST(E2eBunnyCuda, TunedSiftHasMorePointsAndBeatsColmap) {
  // peak_threshold 0.002 + octave_resolution 4 (sweep, shared/findings 2026-11-21): measured
  // 36/36, 5948 points, rotation 0.200 / 0.547 deg, centre 0.35 / 0.94 %.
  sift_run_meets_gate({"bunny_sift_tuned.json", 0.25, 0.65, 1.1, 5000});
}

TEST(E2eBunnyCuda, TunedSift2IsTheMostAccurate) {
  // tuned + peak 0.0015, E-RANSAC 1 px, min 10 inliers, final BA Huber 0.3 px (sweep,
  // shared/findings 2026-11-21): measured 36/36, 5960 points, rotation 0.171 / 0.526 deg,
  // centre 0.33 / 0.91 %.
  sift_run_meets_gate({"bunny_sift_tuned2.json", 0.21, 0.62, 1.05, 5000});
}

// Own images usually come with intrinsics only (and no masks): calib files P = K [I | 0], so every
// true centre is the origin and the ground-truth alignment is undetermined. The run must finish and
// report no pose errors (-1) instead of failing in the evaluation (1.0.0 threw "umeyama: points are
// collinear or coincide").
TEST(E2eBunnyCuda, IntrinsicsOnlyCalibrationReportsNoPoseErrors) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path root = fs::temp_directory_path() / "sfm_e2e_bunny_intrinsics";
  fs::remove_all(root);
  fs::create_directories(root / "images");
  fs::create_directories(root / "calib");
  for (int id = 0; id < 36; ++id) {
    char name[16];
    std::snprintf(name, sizeof name, "%04d", id);
    fs::create_symlink(src / "data" / "bunny" / "images" / (std::string(name) + ".ppm"),
                       root / "images" / (std::string(name) + ".ppm"));
    const Eigen::Matrix3d K = sfm::data::decompose_projection(
                                  sfm::data::read_projection(src / "data" / "bunny" / "calib" /
                                                             (std::string(name) + ".txt")))
                                  .K;
    std::ofstream out(root / "calib" / (std::string(name) + ".txt"));
    out.precision(17);
    out << "CONTOUR\n";
    for (int r = 0; r < 3; ++r) out << K(r, 0) << " " << K(r, 1) << " " << K(r, 2) << " 0\n";
  }
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_sift_tuned2.json", src);
  cfg.dataset.root = root;
  cfg.dataset.masks_dir = "";
  cfg.dataset.use_masks = false;
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_e2e_bunny_intrinsics_out";
  cfg.output.dump_stages = false;
  fs::remove_all(cfg.run.output_dir);
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  const auto r = sfm::pipeline::run_pipeline(cfg, {}, log);
  ASSERT_TRUE(r.metrics.has_value());
  const auto m = r.metrics.value_or(sfm::pipeline::Metrics{});
  EXPECT_GE(m.registered_ids.size(), 3U);
  EXPECT_EQ(m.rotation_median_deg, -1.0);
  EXPECT_EQ(m.centre_max_pct, -1.0);
  const nlohmann::json j = nlohmann::json::parse(read_file(r.run_dir / "metrics.json"));
  EXPECT_EQ(j["rotation_error_deg"]["median"].get<double>(), -1.0);
  fs::remove_all(root);
  fs::remove_all(cfg.run.output_dir);
}
