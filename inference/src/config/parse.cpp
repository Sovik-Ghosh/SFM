// JSON -> Config: structure and types. Semantic checks live in validate.cpp.
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <nlohmann/json.hpp>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"

namespace sfm::config {

namespace {

using nlohmann::json;

// One JSON object at a known path. Every key must be read exactly once; finish() rejects the rest.
class Section {
 public:
  Section(const json& j, std::string path) : j_(j), path_(std::move(path)) {
    if (!j_.is_object()) fail(path_.empty() ? "config" : path_, "expected object", j_);
  }

  std::string str(const std::string& key) {
    const json& v = get(key);
    if (!v.is_string()) fail(at(key), "expected string", v);
    return v.get<std::string>();
  }

  double number(const std::string& key) {
    const json& v = get(key);
    if (!v.is_number()) fail(at(key), "expected number", v);
    return v.get<double>();
  }

  bool boolean(const std::string& key) {
    const json& v = get(key);
    if (!v.is_boolean()) fail(at(key), "expected boolean", v);
    return v.get<bool>();
  }

  template <class Int>
  Int integer(const std::string& key) {
    const json& v = get(key);
    if (!v.is_number_integer()) fail(at(key), "expected integer", v);
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<Int>::max());
    if (v.is_number_unsigned()) {
      const auto u = v.get<std::uint64_t>();
      if (u > kMax) out_of_range<Int>(key, v);
      return static_cast<Int>(u);
    }
    const auto s = v.get<std::int64_t>();
    if constexpr (std::is_signed_v<Int>) {
      if (s < std::numeric_limits<Int>::min() || s > std::numeric_limits<Int>::max()) {
        out_of_range<Int>(key, v);
      }
    } else {
      if (s < 0 || static_cast<std::uint64_t>(s) > kMax) out_of_range<Int>(key, v);
    }
    return static_cast<Int>(s);
  }

  Section object(const std::string& key) { return Section(get(key), at(key)); }

  // For optional keys: true when present (then read it with the typed getters).
  bool has(const std::string& key) const { return j_.contains(key); }

  void finish() const {
    for (const auto& item : j_.items()) {
      if (used_.count(item.key()) == 0) throw ConfigError(at(item.key()) + ": unknown key");
    }
  }

  std::string at(const std::string& key) const { return path_.empty() ? key : path_ + "." + key; }

 private:
  const json& get(const std::string& key) {
    const auto it = j_.find(key);
    if (it == j_.end()) throw ConfigError(at(key) + ": missing");
    used_.insert(key);
    return *it;
  }

  [[noreturn]] static void fail(const std::string& path, const std::string& what, const json& v) {
    throw ConfigError(path + ": " + what + ", got " + v.type_name() + " " + v.dump());
  }

  template <class Int>
  [[noreturn]] void out_of_range(const std::string& key, const json& v) const {
    throw ConfigError(at(key) + ": " + v.dump() + " out of range [" +
                      std::to_string(std::numeric_limits<Int>::min()) + ", " +
                      std::to_string(std::numeric_limits<Int>::max()) + "]");
  }

  const json& j_;
  std::string path_;
  std::set<std::string> used_;
};

}  // namespace

Config parse_config(std::string_view json_text) {
  json root;
  try {
    root = json::parse(json_text);
  } catch (const json::parse_error& e) {
    throw ConfigError(std::string("config: invalid JSON: ") + e.what());
  }

  Section top(root, "");
  Config c;
  c.schema_version = top.integer<int>("schema_version");  // first, so a newer schema says so
  if (c.schema_version >= 1 && c.schema_version < kSchemaVersion) {
    throw ConfigError("schema_version: version " + std::to_string(c.schema_version) +
                      " is no longer read; upgrade the file with tools/migrate_config.py (adds "
                      "the keys of later versions with their defaults)");
  }
  if (c.schema_version != kSchemaVersion) {
    throw ConfigError("schema_version: unsupported version " + std::to_string(c.schema_version) +
                      " (this build reads " + std::to_string(kSchemaVersion) + ")");
  }

  Section run = top.object("run");
  c.run.name = run.str("name");
  c.run.output_dir = run.str("output_dir");
  const std::string level = run.str("log_level");
  try {
    c.run.log_level = core::parse_log_level(level);
  } catch (const ConfigError& e) {
    throw ConfigError(run.at("log_level") + ": " + e.what());
  }
  c.run.seed = run.integer<std::uint64_t>("seed");
  run.finish();

  Section device = top.object("device");
  c.device.index = device.integer<int>("index");
  device.finish();

  Section ds = top.object("dataset");
  c.dataset.root = ds.str("root");
  c.dataset.images_dir = ds.str("images_dir");
  c.dataset.masks_dir = ds.str("masks_dir");
  c.dataset.calib_dir = ds.str("calib_dir");
  c.dataset.first = ds.integer<int>("first");
  c.dataset.last = ds.integer<int>("last");
  c.dataset.use_masks = ds.boolean("use_masks");
  if (ds.has("layout")) c.dataset.layout = ds.str("layout");
  if (ds.has("mask_object")) {
    const std::string mo = ds.str("mask_object");
    if (mo != "black" && mo != "white") {
      throw ConfigError(ds.at("mask_object") + ": expected \"black\" or \"white\", got \"" + mo +
                        "\"");
    }
    c.dataset.mask_object_white = mo == "white";
  }
  if (ds.has("camera")) {
    Section cam = ds.object("camera");
    config::CameraConfig& k = c.dataset.camera;
    if (cam.has("fx")) k.fx = cam.number("fx");
    if (cam.has("fy")) k.fy = cam.number("fy");
    if (cam.has("cx")) k.cx = cam.number("cx");
    if (cam.has("cy")) k.cy = cam.number("cy");
    if (cam.has("exif")) k.exif = cam.boolean("exif");
    if (cam.has("sensor_width_mm")) k.sensor_width_mm = cam.number("sensor_width_mm");
    if (cam.has("fx_default_factor")) k.fx_default_factor = cam.number("fx_default_factor");
    cam.finish();
  }
  if (ds.has("distortion")) {
    Section dist = ds.object("distortion");
    if (dist.has("k1")) c.dataset.k1 = dist.number("k1");
    if (dist.has("k2")) c.dataset.k2 = dist.number("k2");
    dist.finish();
  }
  ds.finish();

  Section mem = top.object("memory");
  c.memory.arena_mb = mem.integer<std::uint32_t>("arena_mb");
  c.memory.max_images = mem.integer<std::uint32_t>("max_images");
  c.memory.max_kpts_per_image = mem.integer<std::uint32_t>("max_kpts_per_image");
  c.memory.max_matches_per_pair = mem.integer<std::uint32_t>("max_matches_per_pair");
  mem.finish();

  Section fe = top.object("features");
  c.features.method = fe.str("method");
  c.features.max_features = fe.integer<std::uint32_t>("max_features");
  c.features.pyramid_levels = fe.integer<int>("pyramid_levels");
  c.features.scale = fe.number("scale");
  c.features.fast_threshold = fe.integer<int>("fast_threshold");
  c.features.edge_border = fe.integer<int>("edge_border");
  Section si = fe.object("sift");
  c.features.sift.octaves = si.integer<int>("octaves");
  c.features.sift.octave_resolution = si.integer<int>("octave_resolution");
  c.features.sift.first_octave = si.integer<int>("first_octave");
  c.features.sift.peak_threshold = si.number("peak_threshold");
  c.features.sift.edge_threshold = si.number("edge_threshold");
  c.features.sift.max_orientations = si.integer<int>("max_orientations");
  si.finish();
  fe.finish();

  Section ma = top.object("matching");
  c.matching.method = ma.str("method");
  c.matching.ratio = ma.number("ratio");
  c.matching.mutual = ma.boolean("mutual");
  Section rs = ma.object("ransac");
  c.matching.ransac.model = rs.str("model");
  c.matching.ransac.threshold_px = rs.number("threshold_px");
  c.matching.ransac.confidence = rs.number("confidence");
  c.matching.ransac.max_iters = rs.integer<std::uint32_t>("max_iters");
  rs.finish();
  Section pr = ma.object("pairs");
  c.matching.pairs.mode = pr.str("mode");
  c.matching.pairs.window = pr.integer<std::uint32_t>("window");
  c.matching.pairs.keyframe_stride = pr.integer<std::uint32_t>("keyframe_stride");
  pr.finish();
  c.matching.min_inliers = ma.integer<std::uint32_t>("min_inliers");
  ma.finish();

  Section re = top.object("reconstruction");
  c.reconstruction.init_min_inliers = re.integer<std::uint32_t>("init_min_inliers");
  c.reconstruction.init_min_angle_deg = re.number("init_min_angle_deg");
  Section pnp = re.object("pnp");
  c.reconstruction.pnp.threshold_px = pnp.number("threshold_px");
  c.reconstruction.pnp.min_inliers = pnp.integer<std::uint32_t>("min_inliers");
  c.reconstruction.pnp.min_inlier_ratio = pnp.number("min_inlier_ratio");
  c.reconstruction.pnp.max_iters = pnp.integer<std::uint32_t>("max_iters");
  pnp.finish();
  c.reconstruction.min_correspondences = re.integer<std::uint32_t>("min_correspondences");
  Section tri = re.object("triangulation");
  c.reconstruction.triangulation.min_angle_deg = tri.number("min_angle_deg");
  c.reconstruction.triangulation.max_reproj_px = tri.number("max_reproj_px");
  tri.finish();
  c.reconstruction.rig = re.boolean("rig");
  re.finish();

  Section ba = top.object("ba");
  c.ba.method = ba.str("method");
  c.ba.huber_px = ba.number("huber_px");
  c.ba.max_iters = ba.integer<std::uint32_t>("max_iters");
  c.ba.every_n_cameras = ba.integer<std::uint32_t>("every_n_cameras");
  c.ba.local_cameras = ba.integer<std::uint32_t>("local_cameras");
  c.ba.global_growth = ba.number("global_growth");
  c.ba.final_huber_px = ba.number("final_huber_px");
  c.ba.final_max_iters = ba.integer<std::uint32_t>("final_max_iters");
  if (ba.has("refine_focal")) c.ba.refine_focal = ba.boolean("refine_focal");
  if (ba.has("refine_distortion")) c.ba.refine_distortion = ba.boolean("refine_distortion");
  ba.finish();

  Section out = top.object("output");
  c.output.dump_stages = out.boolean("dump_stages");
  c.output.ply = out.boolean("ply");
  c.output.colmap = out.boolean("colmap");
  if (out.has("colmap_scene")) c.output.colmap_scene = out.boolean("colmap_scene");
  out.finish();

  top.finish();
  return c;
}

Config load_config(const std::filesystem::path& file, const std::filesystem::path& base_dir) {
  std::ifstream in(file);
  if (!in) throw IoError("config " + file.string() + ": cannot open");
  std::ostringstream text;
  text << in.rdbuf();
  if (in.bad()) throw IoError("config " + file.string() + ": read error");

  Config c = parse_config(text.str());
  const auto absolute = [&](const std::filesystem::path& p) {
    return (p.is_absolute() ? p : base_dir / p).lexically_normal();
  };
  if (!c.run.output_dir.empty()) c.run.output_dir = absolute(c.run.output_dir);
  if (!c.dataset.root.empty()) c.dataset.root = absolute(c.dataset.root);
  validate_config(c, base_dir);
  return c;
}

std::string to_json(const Config& c) {
  nlohmann::ordered_json j;
  j["schema_version"] = c.schema_version;
  j["run"] = {{"name", c.run.name},
              {"output_dir", c.run.output_dir.string()},
              {"log_level", std::string(core::to_string(c.run.log_level))},
              {"seed", c.run.seed}};
  j["device"] = {{"index", c.device.index}};
  j["dataset"] = {{"root", c.dataset.root.string()},
                  {"images_dir", c.dataset.images_dir.string()},
                  {"masks_dir", c.dataset.masks_dir.string()},
                  {"calib_dir", c.dataset.calib_dir.string()},
                  {"first", c.dataset.first},
                  {"last", c.dataset.last},
                  {"use_masks", c.dataset.use_masks}};
  if (c.dataset.layout != "numbered") j["dataset"]["layout"] = c.dataset.layout;
  if (c.dataset.mask_object_white) j["dataset"]["mask_object"] = "white";
  if (c.dataset.camera.enabled()) {
    const CameraConfig& k = c.dataset.camera;
    j["dataset"]["camera"] = {{"fx", k.fx},
                              {"fy", k.fy},
                              {"cx", k.cx},
                              {"cy", k.cy},
                              {"exif", k.exif},
                              {"sensor_width_mm", k.sensor_width_mm},
                              {"fx_default_factor", k.fx_default_factor}};
  }
  j["memory"] = {{"arena_mb", c.memory.arena_mb},
                 {"max_images", c.memory.max_images},
                 {"max_kpts_per_image", c.memory.max_kpts_per_image},
                 {"max_matches_per_pair", c.memory.max_matches_per_pair}};
  j["features"] = {{"method", c.features.method},
                   {"max_features", c.features.max_features},
                   {"pyramid_levels", c.features.pyramid_levels},
                   {"scale", c.features.scale},
                   {"fast_threshold", c.features.fast_threshold},
                   {"edge_border", c.features.edge_border},
                   {"sift",
                    {{"octaves", c.features.sift.octaves},
                     {"octave_resolution", c.features.sift.octave_resolution},
                     {"first_octave", c.features.sift.first_octave},
                     {"peak_threshold", c.features.sift.peak_threshold},
                     {"edge_threshold", c.features.sift.edge_threshold},
                     {"max_orientations", c.features.sift.max_orientations}}}};
  const RansacConfig& rs = c.matching.ransac;
  j["matching"] = {{"method", c.matching.method},
                   {"ratio", c.matching.ratio},
                   {"mutual", c.matching.mutual},
                   {"ransac",
                    {{"model", rs.model},
                     {"threshold_px", rs.threshold_px},
                     {"confidence", rs.confidence},
                     {"max_iters", rs.max_iters}}},
                   {"pairs",
                    {{"mode", c.matching.pairs.mode},
                     {"window", c.matching.pairs.window},
                     {"keyframe_stride", c.matching.pairs.keyframe_stride}}},
                   {"min_inliers", c.matching.min_inliers}};
  const ReconstructionConfig& r = c.reconstruction;
  j["reconstruction"] = {{"init_min_inliers", r.init_min_inliers},
                         {"init_min_angle_deg", r.init_min_angle_deg},
                         {"pnp",
                          {{"threshold_px", r.pnp.threshold_px},
                           {"min_inliers", r.pnp.min_inliers},
                           {"min_inlier_ratio", r.pnp.min_inlier_ratio},
                           {"max_iters", r.pnp.max_iters}}},
                         {"min_correspondences", r.min_correspondences},
                         {"triangulation",
                          {{"min_angle_deg", r.triangulation.min_angle_deg},
                           {"max_reproj_px", r.triangulation.max_reproj_px}}},
                         {"rig", r.rig}};
  j["ba"] = {{"method", c.ba.method},
             {"huber_px", c.ba.huber_px},
             {"max_iters", c.ba.max_iters},
             {"every_n_cameras", c.ba.every_n_cameras},
             {"local_cameras", c.ba.local_cameras},
             {"global_growth", c.ba.global_growth},
             {"final_huber_px", c.ba.final_huber_px},
             {"final_max_iters", c.ba.final_max_iters}};
  if (c.ba.refine_focal) j["ba"]["refine_focal"] = true;
  if (c.ba.refine_distortion) j["ba"]["refine_distortion"] = true;
  if (c.dataset.k1 != 0.0 || c.dataset.k2 != 0.0) {
    j["dataset"]["distortion"] = {{"k1", c.dataset.k1}, {"k2", c.dataset.k2}};
  }
  j["output"] = {
      {"dump_stages", c.output.dump_stages}, {"ply", c.output.ply}, {"colmap", c.output.colmap}};
  if (c.output.colmap_scene) j["output"]["colmap_scene"] = true;
  return j.dump(2) + "\n";
}

}  // namespace sfm::config
