#pragma once
// The CPU pipeline: images -> features -> matching -> tracks -> reconstruction (incremental + BA)
// -> export, with per-stage dumps, metrics.json and timings.json under
// <run.output_dir>/<run.name>/.

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/progress.hpp"
#include "sfm/pipeline/registry.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::pipeline {

enum class Stage : std::uint8_t { kFeatures = 0, kMatching, kTracks, kReconstruction, kExport };

// "features" | "matching" | "tracks" | "reconstruction" | "export"; throws ConfigError otherwise.
Stage parse_stage(const std::string& name);
std::string to_string(Stage s);

struct RunOptions {
  Stage from = Stage::kFeatures;  // earlier stage outputs are loaded from <run>/stages/
  Stage to = Stage::kExport;      // the last stage run
};

struct Metrics {
  int images = 0;
  double keypoints_mean = 0.0;
  int verified_pairs = 0;
  int tracks = 0;
  int init_i = -1;
  int init_j = -1;
  std::vector<int> registered_ids;  // dataset ids of registered images
  int points = 0;
  int observations = 0;
  double reproj_median_px = 0.0;
  double reproj_mean_px = 0.0;
  double reproj_p95_px = 0.0;
  // Against the calibration (ground truth), registered cameras only; -1 when < 3 are registered.
  double rotation_median_deg = -1.0;
  double rotation_max_deg = -1.0;
  double centre_median_pct = -1.0;
  double centre_max_pct = -1.0;
  int pnp_failures = 0;
  int ba_runs = 0;
};

struct Timings {  // wall seconds per stage (0 when the stage was loaded, not run)
  double load = 0.0;
  double features = 0.0;
  double matching = 0.0;
  double tracks = 0.0;
  double reconstruction = 0.0;
  double ba = 0.0;  // part of reconstruction
  double export_ = 0.0;
  double total = 0.0;
};

// The reconstruction as plain arrays (for bindings and library users): one entry per image in the
// range, registered or not, and one per reconstructed point. colours is empty when the export
// stage did not run or wrote nothing (it is read from the images at export).
struct RunModel {
  std::vector<std::string> names;  // image file names
  std::vector<bool> registered;
  std::vector<Eigen::Matrix3d> K;  // pixel centres at integer coordinates
  std::vector<Eigen::Matrix3d> R;  // world -> camera
  std::vector<Eigen::Vector3d> t;
  std::vector<Eigen::Vector3d> points;
  std::vector<std::array<std::uint8_t, 3>> colours;
  double k1 = 0.0;  // radial distortion the model was made with (dataset.distortion or refined)
  double k2 = 0.0;
};

struct RunResult {
  std::filesystem::path run_dir;
  std::optional<Metrics> metrics;  // set when the reconstruction stage ran or was loaded
  Timings timings;
  // Set with metrics.
  std::optional<RunModel> model;
};

// Runs the configured pipeline. Writes <run_dir>/run.log, config.json (the effective config, with
// "auto" methods resolved), stages/ (when output.dump_stages, and always for the stages that a
// later --from needs), metrics.json, timings.json, model.ply, colmap/. Methods named "auto" are
// resolved with resolve_auto_methods(cfg, detect_backend()). `sink` (optional) receives progress
// and may cancel: sfm::Cancelled is thrown at the next check (between stages, images and
// registrations); a run cancelled before the reconstruction ends writes no metrics.json.
// Throws sfm::Error subclasses.
RunResult run_pipeline(const config::Config& cfg, const RunOptions& opt, core::Logger& log,
                       const Registry& registry = default_registry(), ProgressSink* sink = nullptr);

// JSON text of metrics / timings (2-space indent, fixed key order, trailing newline).
std::string to_json(const Metrics& m);
std::string to_json(const Timings& t);

}  // namespace sfm::pipeline
