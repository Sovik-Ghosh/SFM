#pragma once
// A run config for a folder of photos, with the defaults of `sfm_run --images` and Python's
// epipole.reconstruct(): the tuned SIFT config (configs/bunny_sift_tuned2.json), every image in the
// folder ("folder" layout, last = -1), one camera with the focal length from EXIF and refined after
// the final BA, methods "auto" (GPU when usable), and the COLMAP scene export.

#include <filesystem>
#include <string>

#include "sfm/config/config.hpp"

namespace sfm::pipeline {

// images: the folder of photos. masks: optional, <masks>/<photo stem>.png|.pgm, white = object
// unless mask_object_white is false. name: the run name, empty = the folder's name. fx: focal
// length in pixels, 0 = from EXIF. sequential: match neighbouring photos only (video, walk).
struct PhotoOptions {
  std::filesystem::path images;
  std::filesystem::path masks;
  bool mask_object_white = true;
  std::string name;
  std::filesystem::path output_dir = "outputs";
  double fx = 0.0;
  bool sequential = false;
  bool refine_focal = true;
  bool colmap_scene = true;
};

// The template: configs/bunny_sift_tuned2.json as compiled in.
const char* default_config_json();

// The validated config. Relative paths are taken from the current directory. Throws ConfigError
// (or IoError for a missing folder).
config::Config photo_config(const PhotoOptions& o);

}  // namespace sfm::pipeline
