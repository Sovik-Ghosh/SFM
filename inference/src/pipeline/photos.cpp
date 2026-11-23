#include "sfm/pipeline/photos.hpp"

#include <cctype>

#include <nlohmann/json.hpp>

#include "sfm/core/error.hpp"

#include "default_config.hpp"

namespace sfm::pipeline {

namespace fs = std::filesystem;

const char* default_config_json() { return kDefaultConfigJson; }

config::Config photo_config(const PhotoOptions& o) {
  const fs::path cwd = fs::current_path();
  const fs::path images = (o.images.is_absolute() ? o.images : cwd / o.images).lexically_normal();
  if (!fs::is_directory(images)) throw IoError(images.string() + ": not a folder");
  std::string name = o.name;
  if (name.empty()) {
    name = images.filename().string();
    if (name.empty() || name == ".") name = images.parent_path().filename().string();
    for (char& c : name) {  // run names are [A-Za-z0-9._-]
      if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '.' && c != '_' && c != '-') {
        c = '_';
      }
    }
    if (name.empty()) name = "photos";
  }
  nlohmann::json j = nlohmann::json::parse(kDefaultConfigJson);
  j["run"]["name"] = name;
  j["run"]["output_dir"] = o.output_dir.string();
  j["dataset"] = {{"root", images.string()},
                  {"images_dir", "."},
                  {"calib_dir", ""},
                  {"masks_dir", o.masks.empty() ? std::string() : o.masks.string()},
                  {"layout", "folder"},
                  {"first", 0},
                  {"last", -1},
                  {"use_masks", !o.masks.empty()},
                  {"mask_object", o.mask_object_white ? "white" : "black"}};
  if (o.fx > 0.0) {
    j["dataset"]["camera"] = {{"fx", o.fx}};
  } else {
    j["dataset"]["camera"] = {{"exif", true}};
  }
  for (const char* section : {"features", "matching", "ba"}) j[section]["method"] = "auto";
  j["ba"]["refine_focal"] = o.refine_focal;
  if (o.sequential) {
    j["matching"]["pairs"] = {{"mode", "sequential"}, {"window", 10}, {"keyframe_stride", 10}};
  }
  j["output"]["colmap_scene"] = o.colmap_scene;

  config::Config c = config::parse_config(j.dump());
  const auto absolute = [&](const fs::path& p) {
    return (p.is_absolute() ? p : cwd / p).lexically_normal();
  };
  c.run.output_dir = absolute(c.run.output_dir);
  c.dataset.root = absolute(c.dataset.root);
  if (!c.dataset.masks_dir.empty()) c.dataset.masks_dir = absolute(c.dataset.masks_dir);
  config::validate_config(c, cwd);
  return c;
}

}  // namespace sfm::pipeline
