#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace sfm::config::test {

// A valid v5 config (structure and types); the dataset directories are created by the tests.
inline nlohmann::json valid_json() {
  return nlohmann::json::parse(R"({
    "schema_version": 6,
    "run": {"name": "t", "output_dir": "out", "log_level": "debug", "seed": 7},
    "device": {"index": 0},
    "dataset": {"root": "ds", "images_dir": "images", "masks_dir": "silhouettes",
                "calib_dir": "calib", "first": 0, "last": 3, "use_masks": true},
    "memory": {"arena_mb": 16, "max_images": 4, "max_kpts_per_image": 100,
               "max_matches_per_pair": 100},
    "features": {"method": "orb_cpu", "max_features": 50, "pyramid_levels": 3, "scale": 1.25,
                 "fast_threshold": 12, "edge_border": 31,
                 "sift": {"octaves": 3, "octave_resolution": 4, "first_octave": 0,
                          "peak_threshold": 0.01, "edge_threshold": 12.0, "max_orientations": 1}},
    "matching": {"method": "bf_cpu", "ratio": 0.8, "mutual": true,
                 "ransac": {"model": "fundamental", "threshold_px": 2.5, "confidence": 0.99, "max_iters": 500},
                 "pairs": {"mode": "sequential", "window": 5, "keyframe_stride": 4},
                 "min_inliers": 20},
    "reconstruction": {"init_min_inliers": 40, "init_min_angle_deg": 4.0,
                       "pnp": {"threshold_px": 6.0, "min_inliers": 15, "min_inlier_ratio": 0.4, "max_iters": 300},
                       "min_correspondences": 25,
                       "triangulation": {"min_angle_deg": 1.5, "max_reproj_px": 3.0},
                       "rig": false},
    "ba": {"method": "lm_schur_cpu", "huber_px": 1.5, "max_iters": 30, "every_n_cameras": 5,
           "local_cameras": 8, "global_growth": 0.5, "final_huber_px": 0.75, "final_max_iters": 7},
    "output": {"dump_stages": false, "ply": true, "colmap": false}
  })");
}

// "/dataset/first" -> "dataset.first"
inline std::string dotted(const std::string& pointer) {
  std::string s = pointer.substr(1);
  for (char& ch : s) {
    if (ch == '/') ch = '.';
  }
  return s;
}

}  // namespace sfm::config::test
