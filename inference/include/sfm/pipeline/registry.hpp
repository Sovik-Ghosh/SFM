#pragma once
// Method registry: the config names a method per stage ("features.method": "orb_cpu"); the
// pipeline looks it up here. GPU phases register their methods next to the CPU ones, so a run
// switches implementation by config only, and a stage never includes another stage's internals.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "sfm/ba/ba.hpp"
#include "sfm/config/config.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/features/features.hpp"
#include "sfm/matching/matches.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::pipeline {

using FeatureMethod = std::function<features::Features(
    const data::Image8& gray, const data::Image8* mask, const config::FeaturesConfig& cfg)>;
// K: calibrated intrinsics per image (used by matching.ransac.model "essential").
using MatchMethod = std::function<std::vector<matching::PairMatches>(
    const std::vector<features::Features>& features, const config::MatchingConfig& cfg,
    const std::vector<Eigen::Matrix3d>& K, std::uint64_t seed)>;
// free_images: empty = global BA, else a local BA of those images (ba::bundle_adjust).
using BaMethod =
    std::function<ba::Report(recon::Reconstruction& rec, const recon::TrackIndex& index,
                             const config::BaConfig& cfg, const std::vector<int>& free_images)>;

class Registry {
 public:
  void add_features(const std::string& name, FeatureMethod m) { features_[name] = std::move(m); }
  void add_matching(const std::string& name, MatchMethod m) { matching_[name] = std::move(m); }
  void add_ba(const std::string& name, BaMethod m) { ba_[name] = std::move(m); }

  // Throw ConfigError "features.method: unknown method 'x' (known: a, b)" for unknown names.
  const FeatureMethod& features(const std::string& name) const;
  const MatchMethod& matching(const std::string& name) const;
  const BaMethod& ba(const std::string& name) const;

 private:
  std::map<std::string, FeatureMethod> features_;
  std::map<std::string, MatchMethod> matching_;
  std::map<std::string, BaMethod> ba_;
};

// Which implementation "auto" picks: the GPU methods when a CUDA device is visible (driver
// present, not hidden by CUDA_VISIBLE_DEVICES), the CPU methods otherwise.
struct Backend {
  bool gpu = false;
  std::string device;  // name of CUDA device 0, empty without a GPU
};
Backend detect_backend();

// cfg with every method "auto" replaced: orb_cuda / bf_cuda / lm_schur_cuda if gpu, else orb_cpu /
// bf_cpu / lm_schur_cpu. Explicit method names are kept.
config::Config resolve_auto_methods(const config::Config& cfg, bool gpu);

// cfg on one backend: "auto" resolved, then every *_cuda method replaced by its *_cpu twin
// (gpu false) or every *_cpu method by its *_cuda twin (gpu true). sfm_run --backend; the
// pipeline's fallback when a config asks for CUDA methods without a GPU or without CUDA support.
config::Config use_backend(const config::Config& cfg, bool gpu);

// True if this build has the CUDA methods (SFM_WITH_CUDA).
bool built_with_cuda();

// The built-in methods: features "orb_cpu", "orb_cuda" (GPU, bit-identical output), matching
// "bf_cpu", "bf_cuda" (GPU top-2 identical, GPU RANSAC with every hypothesis scored), ba
// "lm_schur_cpu", "lm_schur_cuda" (GPU linear algebra, same LM loop). Not thread-safe: orb_cuda
// keeps its device buffers between calls.
const Registry& default_registry();

}  // namespace sfm::pipeline
