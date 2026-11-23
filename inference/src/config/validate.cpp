// Semantic checks on a parsed Config. Collects every problem so one run reports them all.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"

namespace sfm::config {

namespace {

namespace fs = std::filesystem;

bool is_safe_name(const std::string& s) {
  return !s.empty() && s != "." && s != ".." && std::all_of(s.begin(), s.end(), [](char ch) {
    const auto c = static_cast<unsigned char>(ch);
    return std::isalnum(c) || c == '.' || c == '_' || c == '-';
  });
}

class Problems {
 public:
  void add(const std::string& path, const std::string& what) {
    list_.push_back(path + ": " + what);
  }

  void require_dir(const std::string& path, const fs::path& dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) add(path, "directory not found: " + dir.string());
  }

  void throw_if_any() const {
    if (list_.empty()) return;
    std::string msg = "invalid config (" + std::to_string(list_.size()) + " problem" +
                      (list_.size() == 1 ? "" : "s") + "): ";
    for (std::size_t i = 0; i < list_.size(); ++i) msg += (i ? "; " : "") + list_[i];
    throw ConfigError(msg);
  }

 private:
  std::vector<std::string> list_;
};

}  // namespace

void validate_config(const Config& c, const fs::path& base_dir) {
  Problems p;

  if (c.schema_version != kSchemaVersion) {
    p.add("schema_version", "must be " + std::to_string(kSchemaVersion));
  }
  if (!is_safe_name(c.run.name)) p.add("run.name", "must match [A-Za-z0-9._-]+");
  if (c.run.output_dir.empty()) p.add("run.output_dir", "must not be empty");
  if (c.device.index < 0) p.add("device.index", "must be >= 0");

  const DatasetConfig& d = c.dataset;
  const bool folder = d.layout == "folder";
  if (d.layout != "numbered" && !folder) {
    p.add("dataset.layout", "must be \"numbered\" or \"folder\"");
  }
  if (d.first < 0) p.add("dataset.first", "must be >= 0");
  if (folder && d.last != -1 && d.last < d.first) {
    p.add("dataset.last", "must be >= dataset.first, or -1 for all images");
  }
  if (!folder && d.last < d.first) p.add("dataset.last", "must be >= dataset.first");
  if (!folder && d.last > 9999) p.add("dataset.last", "must be <= 9999 (NNNN file names)");
  const CameraConfig& k = d.camera;
  if (k.fx < 0.0) p.add("dataset.camera.fx", "must be > 0");
  if (k.fy < 0.0) p.add("dataset.camera.fy", "must be >= 0 (0: same as fx)");
  if (k.fy > 0.0 && k.fx <= 0.0) p.add("dataset.camera.fx", "must be set when fy is");
  if (k.exif && k.fx > 0.0) p.add("dataset.camera", "set either fx or exif, not both");
  if (k.sensor_width_mm < 0.0) p.add("dataset.camera.sensor_width_mm", "must be >= 0");
  if (k.fx_default_factor <= 0.0) p.add("dataset.camera.fx_default_factor", "must be > 0");
  if (d.root.empty()) {
    p.add("dataset.root", "must not be empty");
  } else {
    const fs::path root = d.root.is_absolute() ? d.root : base_dir / d.root;
    p.require_dir("dataset.root", root);
    p.require_dir("dataset.images_dir", root / d.images_dir);
    if (!k.enabled()) p.require_dir("dataset.calib_dir", root / d.calib_dir);
    if (d.use_masks) {
      // An empty masks_dir would resolve to the dataset root and fail later at load time.
      if (d.masks_dir.empty()) {
        p.add("dataset.masks_dir", "must not be empty when use_masks is true");
      } else {
        p.require_dir("dataset.masks_dir", root / d.masks_dir);
      }
    }
  }

  const MemoryConfig& m = c.memory;
  if (m.arena_mb == 0) p.add("memory.arena_mb", "must be > 0");
  if (m.max_kpts_per_image == 0) p.add("memory.max_kpts_per_image", "must be > 0");
  if (m.max_matches_per_pair == 0) p.add("memory.max_matches_per_pair", "must be > 0");
  if (d.last >= d.first && m.max_images < static_cast<unsigned>(d.image_count())) {
    p.add("memory.max_images",
          "must be >= the number of images (" + std::to_string(d.image_count()) + ")");
  }

  const FeaturesConfig& f = c.features;
  if (f.method.empty()) p.add("features.method", "must not be empty");
  if (f.max_features == 0) p.add("features.max_features", "must be > 0");
  if (f.max_features > m.max_kpts_per_image) {
    p.add("features.max_features",
          "must be <= memory.max_kpts_per_image (" + std::to_string(m.max_kpts_per_image) + ")");
  }
  if (f.pyramid_levels < 1 || f.pyramid_levels > 16) {
    p.add("features.pyramid_levels", "must be in [1, 16]");
  }
  if (!(f.scale > 1.0 && f.scale <= 2.0)) p.add("features.scale", "must be in (1, 2]");
  if (f.fast_threshold < 1 || f.fast_threshold > 254) {
    p.add("features.fast_threshold", "must be in [1, 254]");
  }
  if (f.edge_border < 22) p.add("features.edge_border", "must be >= 22 (descriptor patch radius)");
  const SiftConfig& s = f.sift;
  if (s.octaves < 1 || s.octaves > 8) p.add("features.sift.octaves", "must be in [1, 8]");
  if (s.octave_resolution < 1 || s.octave_resolution > 6) {
    p.add("features.sift.octave_resolution", "must be in [1, 6]");
  }
  if (s.first_octave != -1 && s.first_octave != 0) {
    p.add("features.sift.first_octave", "must be -1 or 0");
  }
  if (!(s.peak_threshold > 0.0)) p.add("features.sift.peak_threshold", "must be > 0");
  if (!(s.edge_threshold > 1.0)) p.add("features.sift.edge_threshold", "must be > 1");
  if (s.max_orientations < 1 || s.max_orientations > 4) {
    p.add("features.sift.max_orientations", "must be in [1, 4]");
  }
  if (f.method.rfind("sift", 0) == 0 && s.max_orientations >= 1 &&
      static_cast<std::uint64_t>(f.max_features) * static_cast<std::uint64_t>(s.max_orientations) >
          m.max_kpts_per_image) {
    p.add("features.max_features",
          "x features.sift.max_orientations must be <= memory.max_kpts_per_image (" +
              std::to_string(m.max_kpts_per_image) + ")");
  }

  const MatchingConfig& mc = c.matching;
  if (mc.method.empty()) p.add("matching.method", "must not be empty");
  if (!(mc.ratio > 0.0 && mc.ratio <= 1.0)) p.add("matching.ratio", "must be in (0, 1]");
  if (mc.pairs.mode != "exhaustive" && mc.pairs.mode != "sequential") {
    p.add("matching.pairs.mode", "must be \"exhaustive\" or \"sequential\"");
  }
  if (mc.pairs.mode == "sequential" && mc.pairs.window == 0) {
    p.add("matching.pairs.window", "must be > 0 for sequential pairs");
  }
  if (mc.ransac.model != "essential" && mc.ransac.model != "fundamental") {
    p.add("matching.ransac.model", "must be \"essential\" or \"fundamental\"");
  }
  if (!(mc.ransac.threshold_px > 0.0)) p.add("matching.ransac.threshold_px", "must be > 0");
  if (!(mc.ransac.confidence > 0.0 && mc.ransac.confidence < 1.0)) {
    p.add("matching.ransac.confidence", "must be in (0, 1)");
  }
  if (mc.ransac.max_iters == 0) p.add("matching.ransac.max_iters", "must be > 0");
  if (mc.min_inliers < 8) p.add("matching.min_inliers", "must be >= 8 (8-point minimum)");
  if (mc.min_inliers > m.max_matches_per_pair) {
    p.add("matching.min_inliers", "must be <= memory.max_matches_per_pair");
  }

  const ReconstructionConfig& r = c.reconstruction;
  if (r.init_min_inliers < 8) p.add("reconstruction.init_min_inliers", "must be >= 8");
  if (!(r.init_min_angle_deg >= 0.0 && r.init_min_angle_deg < 90.0)) {
    p.add("reconstruction.init_min_angle_deg", "must be in [0, 90)");
  }
  if (!(r.pnp.threshold_px > 0.0)) p.add("reconstruction.pnp.threshold_px", "must be > 0");
  if (r.pnp.min_inliers < 6) p.add("reconstruction.pnp.min_inliers", "must be >= 6 (DLT minimum)");
  if (r.pnp.max_iters == 0) p.add("reconstruction.pnp.max_iters", "must be > 0");
  if (!(r.pnp.min_inlier_ratio >= 0.0 && r.pnp.min_inlier_ratio <= 1.0)) {
    p.add("reconstruction.pnp.min_inlier_ratio", "must be in [0, 1]");
  }
  if (r.min_correspondences < r.pnp.min_inliers) {
    p.add("reconstruction.min_correspondences", "must be >= reconstruction.pnp.min_inliers");
  }
  if (!(r.triangulation.min_angle_deg >= 0.0 && r.triangulation.min_angle_deg < 90.0)) {
    p.add("reconstruction.triangulation.min_angle_deg", "must be in [0, 90)");
  }
  if (!(r.triangulation.max_reproj_px > 0.0)) {
    p.add("reconstruction.triangulation.max_reproj_px", "must be > 0");
  }

  if (c.ba.method.empty()) p.add("ba.method", "must not be empty");
  if (!(c.ba.huber_px > 0.0)) p.add("ba.huber_px", "must be > 0");
  if (!(c.ba.global_growth > 0.0 && c.ba.global_growth <= 10.0)) {
    p.add("ba.global_growth", "must be in (0, 10]");
  }
  if (!(c.ba.final_huber_px > 0.0)) p.add("ba.final_huber_px", "must be > 0");
  if (c.ba.refine_focal && c.ba.final_max_iters == 0) {
    p.add("ba.refine_focal", "needs the final BA (ba.final_max_iters > 0)");
  }
  if (c.ba.refine_distortion && c.ba.final_max_iters == 0) {
    p.add("ba.refine_distortion", "needs the final BA (ba.final_max_iters > 0)");
  }
  if (!(std::abs(c.dataset.k1) < 1.0)) p.add("dataset.distortion.k1", "must be in (-1, 1)");
  if (!(std::abs(c.dataset.k2) < 1.0)) p.add("dataset.distortion.k2", "must be in (-1, 1)");

  p.throw_if_any();
}

}  // namespace sfm::config
