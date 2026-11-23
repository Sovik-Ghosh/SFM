#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"

#include "config_test_util.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using sfm::ConfigError;
using sfm::config::parse_config;
using sfm::config::test::dotted;
using sfm::config::test::valid_json;

namespace {

// Expects parse_config(j) to throw ConfigError whose message starts with `prefix`.
void expect_error(const json& j, const std::string& prefix) {
  try {
    parse_config(j.dump());
    ADD_FAILURE() << "no ConfigError; expected \"" << prefix << "...\"";
  } catch (const ConfigError& e) {
    EXPECT_EQ(std::string(e.what()).rfind(prefix, 0), 0u) << e.what();
  }
}

}  // namespace

TEST(ConfigParse, ReadsEveryField) {
  const auto c = parse_config(valid_json().dump());
  EXPECT_EQ(c.schema_version, 6);
  EXPECT_EQ(c.features.sift.octaves, 3);
  EXPECT_EQ(c.features.sift.octave_resolution, 4);
  EXPECT_EQ(c.features.sift.first_octave, 0);
  EXPECT_EQ(c.features.sift.peak_threshold, 0.01);
  EXPECT_EQ(c.features.sift.edge_threshold, 12.0);
  EXPECT_EQ(c.features.sift.max_orientations, 1);
  EXPECT_EQ(c.run.name, "t");
  EXPECT_EQ(c.run.output_dir, fs::path("out"));
  EXPECT_EQ(c.run.log_level, sfm::core::LogLevel::kDebug);
  EXPECT_EQ(c.run.seed, 7u);
  EXPECT_EQ(c.device.index, 0);
  EXPECT_EQ(c.dataset.root, fs::path("ds"));
  EXPECT_EQ(c.dataset.images_dir, fs::path("images"));
  EXPECT_EQ(c.dataset.masks_dir, fs::path("silhouettes"));
  EXPECT_EQ(c.dataset.calib_dir, fs::path("calib"));
  EXPECT_EQ(c.dataset.first, 0);
  EXPECT_EQ(c.dataset.last, 3);
  EXPECT_EQ(c.dataset.image_count(), 4);
  EXPECT_TRUE(c.dataset.use_masks);
  EXPECT_EQ(c.memory.arena_mb, 16u);
  EXPECT_EQ(c.memory.max_images, 4u);
  EXPECT_EQ(c.memory.max_kpts_per_image, 100u);
  EXPECT_EQ(c.memory.max_matches_per_pair, 100u);
  EXPECT_EQ(c.features.method, "orb_cpu");
  EXPECT_EQ(c.features.max_features, 50u);
  EXPECT_EQ(c.features.pyramid_levels, 3);
  EXPECT_DOUBLE_EQ(c.features.scale, 1.25);
  EXPECT_EQ(c.features.fast_threshold, 12);
  EXPECT_EQ(c.features.edge_border, 31);
  EXPECT_EQ(c.matching.method, "bf_cpu");
  EXPECT_DOUBLE_EQ(c.matching.ratio, 0.8);
  EXPECT_TRUE(c.matching.mutual);
  EXPECT_EQ(c.matching.ransac.model, "fundamental");
  EXPECT_EQ(c.matching.pairs.mode, "sequential");
  EXPECT_EQ(c.matching.pairs.window, 5u);
  EXPECT_EQ(c.matching.pairs.keyframe_stride, 4u);
  EXPECT_DOUBLE_EQ(c.matching.ransac.threshold_px, 2.5);
  EXPECT_DOUBLE_EQ(c.matching.ransac.confidence, 0.99);
  EXPECT_EQ(c.matching.ransac.max_iters, 500u);
  EXPECT_EQ(c.matching.min_inliers, 20u);
  EXPECT_EQ(c.reconstruction.init_min_inliers, 40u);
  EXPECT_DOUBLE_EQ(c.reconstruction.init_min_angle_deg, 4.0);
  EXPECT_DOUBLE_EQ(c.reconstruction.pnp.threshold_px, 6.0);
  EXPECT_EQ(c.reconstruction.pnp.min_inliers, 15u);
  EXPECT_DOUBLE_EQ(c.reconstruction.pnp.min_inlier_ratio, 0.4);
  EXPECT_EQ(c.reconstruction.pnp.max_iters, 300u);
  EXPECT_EQ(c.reconstruction.min_correspondences, 25u);
  EXPECT_DOUBLE_EQ(c.reconstruction.triangulation.min_angle_deg, 1.5);
  EXPECT_DOUBLE_EQ(c.reconstruction.triangulation.max_reproj_px, 3.0);
  EXPECT_EQ(c.ba.method, "lm_schur_cpu");
  EXPECT_DOUBLE_EQ(c.ba.huber_px, 1.5);
  EXPECT_EQ(c.ba.max_iters, 30u);
  EXPECT_EQ(c.ba.every_n_cameras, 5u);
  EXPECT_EQ(c.ba.local_cameras, 8u);
  EXPECT_DOUBLE_EQ(c.ba.global_growth, 0.5);
  EXPECT_DOUBLE_EQ(c.ba.final_huber_px, 0.75);
  EXPECT_EQ(c.ba.final_max_iters, 7u);
  EXPECT_FALSE(c.output.dump_stages);
  EXPECT_TRUE(c.output.ply);
  EXPECT_FALSE(c.output.colmap);
}

TEST(ConfigParse, EveryMissingKeyIsNamed) {
  const json flat = valid_json().flatten();
  for (const auto& leaf : flat.items()) {
    SCOPED_TRACE(leaf.key());
    json j = valid_json();
    const json::json_pointer ptr(leaf.key());
    j[ptr.parent_pointer()].erase(ptr.back());
    expect_error(j, dotted(leaf.key()) + ": missing");
  }
  for (const char* section : {"run", "device", "dataset", "memory", "features", "matching",
                              "reconstruction", "ba", "output"}) {
    json j = valid_json();
    j.erase(section);
    expect_error(j, std::string(section) + ": missing");
  }
}

TEST(ConfigParse, EveryWrongTypeIsNamed) {
  const json flat = valid_json().flatten();
  for (const auto& leaf : flat.items()) {
    SCOPED_TRACE(leaf.key());
    json j = valid_json();
    j[json::json_pointer(leaf.key())] = json::array();
    expect_error(j, dotted(leaf.key()) + ": expected ");
  }
  json j = valid_json();
  j["memory"] = 5;
  expect_error(j, "memory: expected object");
}

TEST(ConfigParse, UnknownKeysAreRejectedAtEveryLevel) {
  for (const char* level :
       {"", "/run", "/device", "/dataset", "/memory", "/features", "/features/sift", "/matching",
        "/matching/ransac", "/matching/pairs", "/reconstruction", "/reconstruction/pnp",
        "/reconstruction/triangulation", "/ba", "/output"}) {
    const std::string where = level;
    SCOPED_TRACE(where);
    json j = valid_json();
    j[json::json_pointer(where)]["typo"] = 1;
    expect_error(j, (where.empty() ? "" : dotted(where) + ".") + "typo: unknown key");
  }
}

TEST(ConfigParse, OtherSchemaVersionsAreRejected) {
  json j = valid_json();
  j["schema_version"] = 7;
  j["new_section"] = json::object();  // a v7 file: the version message wins over unknown keys
  expect_error(j, "schema_version: unsupported version 7");
}

TEST(ConfigParse, Version1PointsToTheMigrationTool) {
  json j = valid_json();
  j["schema_version"] = 1;
  for (const char* s : {"features", "matching", "reconstruction", "ba", "output"}) j.erase(s);
  expect_error(j,
               "schema_version: version 1 is no longer read; upgrade the file with "
               "tools/migrate_config.py");
  j = valid_json();
  j["schema_version"] = 2;
  j["matching"]["ransac"].erase("model");
  expect_error(j, "schema_version: version 2 is no longer read");
  j = valid_json();
  j["schema_version"] = 3;
  j["features"].erase("sift");
  expect_error(j, "schema_version: version 3 is no longer read");
  j = valid_json();
  j["schema_version"] = 4;
  j["ba"].erase("local_cameras");
  j["ba"].erase("global_growth");
  j["ba"].erase("final_huber_px");
  j["ba"].erase("final_max_iters");
  expect_error(j, "schema_version: version 4 is no longer read");
  j = valid_json();
  j["schema_version"] = 5;
  j["reconstruction"].erase("rig");
  expect_error(j, "schema_version: version 5 is no longer read");
}

TEST(ConfigParse, NumbersAcceptIntegers) {
  json j = valid_json();
  j["features"]["scale"] = 2;  // a JSON integer is a valid number
  EXPECT_DOUBLE_EQ(parse_config(j.dump()).features.scale, 2.0);
  j["features"]["scale"] = "1.2";
  expect_error(j, "features.scale: expected number, got string");
}

TEST(ConfigParse, IntegerRangesAndKinds) {
  json j = valid_json();
  j["memory"]["arena_mb"] = -1;
  expect_error(j, "memory.arena_mb: -1 out of range [0, 4294967295]");
  j["memory"]["arena_mb"] = 5000000000ULL;
  expect_error(j, "memory.arena_mb: 5000000000 out of range");
  j = valid_json();
  j["dataset"]["first"] = 1.5;
  expect_error(j, "dataset.first: expected integer, got number 1.5");
  j["dataset"]["first"] = 3000000000ULL;
  expect_error(j, "dataset.first: 3000000000 out of range");
  j = valid_json();
  j["run"]["seed"] = 18446744073709551615ULL;
  EXPECT_EQ(parse_config(j.dump()).run.seed, 18446744073709551615ULL);
}

TEST(ConfigParse, BadLogLevelAndBadJson) {
  json j = valid_json();
  j["run"]["log_level"] = "verbose";
  expect_error(j, "run.log_level: log level must be debug|info|warn|error");
  EXPECT_THROW(parse_config("{\"schema_version\": 1,"), ConfigError);
  EXPECT_THROW(parse_config(""), ConfigError);
  EXPECT_THROW(parse_config("[1, 2]"), ConfigError);
}

TEST(ConfigParse, ToJsonRoundTrips) {
  const auto c = parse_config(valid_json().dump());
  const std::string text = sfm::config::to_json(c);
  EXPECT_EQ(json::parse(text), valid_json());
  EXPECT_EQ(sfm::config::to_json(parse_config(text)), text);
  EXPECT_EQ(text.back(), '\n');
}

TEST(ConfigParse, OptionalDatasetAndOutputKeys) {
  // Absent: the defaults (configs written before these keys existed still parse).
  const auto d = parse_config(valid_json().dump());
  EXPECT_EQ(d.dataset.layout, "numbered");
  EXPECT_FALSE(d.dataset.mask_object_white);
  EXPECT_FALSE(d.dataset.camera.enabled());
  EXPECT_FALSE(d.output.colmap_scene);

  json j = valid_json();
  j["dataset"]["layout"] = "folder";
  j["dataset"]["mask_object"] = "white";
  j["dataset"]["camera"] = {{"fx", 1500}, {"cy", 400.5}};
  j["output"]["colmap_scene"] = true;
  const auto c = parse_config(j.dump());
  EXPECT_EQ(c.dataset.layout, "folder");
  EXPECT_TRUE(c.dataset.mask_object_white);
  EXPECT_DOUBLE_EQ(c.dataset.camera.fx, 1500.0);
  EXPECT_DOUBLE_EQ(c.dataset.camera.fy, 0.0);
  EXPECT_DOUBLE_EQ(c.dataset.camera.cy, 400.5);
  EXPECT_LT(c.dataset.camera.cx, 0.0);
  EXPECT_TRUE(c.output.colmap_scene);
  // to_json keeps them, so a run's config.json reproduces the run.
  const auto again = parse_config(sfm::config::to_json(c));
  EXPECT_EQ(again.dataset.layout, "folder");
  EXPECT_TRUE(again.dataset.mask_object_white);
  EXPECT_DOUBLE_EQ(again.dataset.camera.fx, 1500.0);
  EXPECT_TRUE(again.output.colmap_scene);

  j["dataset"]["mask_object"] = "grey";
  expect_error(j, "dataset.mask_object: expected \"black\" or \"white\"");
  j = valid_json();
  j["dataset"]["camera"] = {{"focal", 1500}};
  expect_error(j, "dataset.camera.focal: unknown key");
  j["dataset"]["camera"] = {{"exif", "yes"}};
  expect_error(j, "dataset.camera.exif: expected boolean");
}
