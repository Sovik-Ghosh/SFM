#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"

#include "config_test_util.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using sfm::ConfigError;
using sfm::config::Config;
using sfm::config::load_config;
using sfm::config::parse_config;
using sfm::config::validate_config;
using sfm::config::test::valid_json;

namespace {

// Temp base dir holding ds/{images,silhouettes,calib} to match valid_json().
class ConfigValidate : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    base_ = fs::temp_directory_path() / (std::string("sfm_config_") + info->name());
    for (const char* d : {"images", "silhouettes", "calib"}) {
      fs::create_directories(base_ / "ds" / d);
    }
  }
  void TearDown() override { fs::remove_all(base_); }

  Config parsed(const json& j) const { return parse_config(j.dump()); }

  // Validation message, or "" if valid.
  std::string problems(const json& j) const {
    try {
      validate_config(parsed(j), base_);
    } catch (const ConfigError& e) {
      return e.what();
    }
    return "";
  }

  fs::path base_;
};

}  // namespace

TEST_F(ConfigValidate, ValidConfigPasses) { EXPECT_EQ(problems(valid_json()), ""); }

TEST_F(ConfigValidate, ReportsEveryProblemAtOnce) {
  json j = valid_json();
  j["run"]["name"] = "../escape";
  j["device"]["index"] = -1;
  j["dataset"]["first"] = 5;  // > last
  j["memory"]["arena_mb"] = 0;
  const std::string msg = problems(j);
  EXPECT_NE(msg.find("(4 problems)"), std::string::npos) << msg;
  for (const char* path : {"run.name:", "device.index:", "dataset.last:", "memory.arena_mb:"}) {
    EXPECT_NE(msg.find(path), std::string::npos) << path << " not in: " << msg;
  }
}

TEST_F(ConfigValidate, RunNameMustBeASafeDirectoryName) {
  for (const char* bad : {"", ".", "..", "a/b", "a b", "x\\y"}) {
    json j = valid_json();
    j["run"]["name"] = bad;
    EXPECT_NE(problems(j).find("run.name:"), std::string::npos) << bad;
  }
  json j = valid_json();
  j["run"]["name"] = "bunny-baseline_v1.2";
  EXPECT_EQ(problems(j), "");
}

TEST_F(ConfigValidate, DatasetDirectoriesMustExist) {
  json j = valid_json();
  j["dataset"]["root"] = "nope";
  EXPECT_NE(problems(j).find("dataset.root: directory not found"), std::string::npos);

  fs::remove(base_ / "ds" / "calib");
  EXPECT_NE(problems(valid_json()).find("dataset.calib_dir: directory not found"),
            std::string::npos);
}

TEST_F(ConfigValidate, EmptyMasksDirWithMasksIsRejected) {
  json j = valid_json();
  j["dataset"]["masks_dir"] = "";
  EXPECT_NE(problems(j).find("dataset.masks_dir: must not be empty when use_masks is true"),
            std::string::npos);
  j["dataset"]["use_masks"] = false;
  EXPECT_EQ(problems(j), "");
}

TEST_F(ConfigValidate, MasksDirOnlyNeededWhenUsed) {
  fs::remove(base_ / "ds" / "silhouettes");
  EXPECT_NE(problems(valid_json()).find("dataset.masks_dir:"), std::string::npos);
  json j = valid_json();
  j["dataset"]["use_masks"] = false;
  EXPECT_EQ(problems(j), "");
}

TEST_F(ConfigValidate, MemoryCapsMustFitTheDataset) {
  json j = valid_json();
  j["memory"]["max_images"] = 3;  // dataset has 4
  EXPECT_NE(problems(j).find("memory.max_images: must be >= the number of images (4)"),
            std::string::npos);
  j = valid_json();
  j["memory"]["max_kpts_per_image"] = 0;
  j["memory"]["max_matches_per_pair"] = 0;
  const std::string msg = problems(j);
  EXPECT_NE(msg.find("memory.max_kpts_per_image:"), std::string::npos);
  EXPECT_NE(msg.find("memory.max_matches_per_pair:"), std::string::npos);
}

TEST_F(ConfigValidate, SiftKeypointsWithOrientationsMustFitTheMemoryCap) {
  json j = valid_json();
  j["features"]["method"] = "sift_cpu";
  j["features"]["sift"]["max_orientations"] = 2;  // 50 locations x 2 = 100: fits
  EXPECT_EQ(problems(j), "");
  j["features"]["sift"]["max_orientations"] = 3;  // 150 > 100
  EXPECT_NE(problems(j).find("features.max_features: x features.sift.max_orientations"),
            std::string::npos);
}

TEST_F(ConfigValidate, StageSectionsAreRangeChecked) {
  json j = valid_json();
  j["features"]["max_features"] = 101;  // > memory.max_kpts_per_image (100)
  j["features"]["pyramid_levels"] = 0;
  j["features"]["scale"] = 1.0;
  j["features"]["fast_threshold"] = 0;
  j["features"]["edge_border"] = 10;
  j["matching"]["ratio"] = 1.5;
  j["matching"]["ransac"]["threshold_px"] = 0.0;
  j["matching"]["ransac"]["confidence"] = 1.0;
  j["matching"]["min_inliers"] = 7;
  j["reconstruction"]["init_min_angle_deg"] = 90.0;
  j["reconstruction"]["pnp"]["min_inliers"] = 5;
  j["reconstruction"]["triangulation"]["max_reproj_px"] = -1.0;
  j["ba"]["huber_px"] = 0.0;
  j["ba"]["global_growth"] = 0.0;
  j["ba"]["final_huber_px"] = 0.0;
  j["matching"]["ransac"]["model"] = "homography";
  j["matching"]["pairs"]["mode"] = "random";
  j["reconstruction"]["pnp"]["min_inlier_ratio"] = 1.5;
  j["features"]["sift"]["first_octave"] = 1;
  j["features"]["sift"]["max_orientations"] = 0;
  const std::string msg = problems(j);
  EXPECT_NE(msg.find("(20 problems)"), std::string::npos) << msg;
  for (const char* path : {"features.max_features: must be <= memory.max_kpts_per_image (100)",
                           "features.pyramid_levels:",
                           "features.scale:",
                           "features.fast_threshold:",
                           "features.edge_border:",
                           "matching.ratio:",
                           "matching.ransac.threshold_px:",
                           "matching.ransac.confidence:",
                           "matching.min_inliers:",
                           "reconstruction.init_min_angle_deg:",
                           "reconstruction.pnp.min_inliers:",
                           "reconstruction.triangulation.max_reproj_px:",
                           "ba.huber_px:",
                           "matching.ransac.model:",
                           "reconstruction.pnp.min_inlier_ratio:",
                           "matching.pairs.mode:",
                           "features.sift.first_octave:",
                           "features.sift.max_orientations:",
                           "ba.global_growth:",
                           "ba.final_huber_px:"}) {
    EXPECT_NE(msg.find(path), std::string::npos) << path << " not in: " << msg;
  }
  j = valid_json();
  j["reconstruction"]["min_correspondences"] = 10;  // < pnp.min_inliers (15)
  EXPECT_NE(problems(j).find("reconstruction.min_correspondences:"), std::string::npos);
}

TEST_F(ConfigValidate, LoadConfigResolvesPathsAgainstBase) {
  const fs::path file = base_ / "run.json";
  std::ofstream(file) << valid_json().dump();
  const Config c = load_config(file, base_);
  EXPECT_EQ(c.dataset.root, base_ / "ds");
  EXPECT_EQ(c.run.output_dir, base_ / "out");
  EXPECT_THROW(load_config(base_ / "missing.json", base_), sfm::IoError);
}

TEST(ConfigBunny, RepositoryConfigLoads) {
  const fs::path src(SFM_SOURCE_DIR);
  const Config c = load_config(src / "configs" / "bunny.json", src);
  EXPECT_EQ(c.run.name, "bunny-baseline");
  EXPECT_EQ(c.dataset.root, src / "data" / "bunny");
  EXPECT_EQ(c.dataset.image_count(), 36);
  EXPECT_EQ(c.memory.arena_mb, 512u);
  EXPECT_EQ(c.features.max_features, 2000u);
  EXPECT_EQ(c.matching.min_inliers, 30u);
}

TEST_F(ConfigValidate, FolderLayoutAndSharedCamera) {
  json j = valid_json();
  j["dataset"]["layout"] = "folder";
  j["dataset"]["last"] = -1;  // all images in the folder
  EXPECT_EQ(problems(j), "");
  j["dataset"]["last"] = -2;
  EXPECT_NE(problems(j).find("dataset.last: must be >= dataset.first, or -1"), std::string::npos);
  j["dataset"]["layout"] = "flat";
  EXPECT_NE(problems(j).find("dataset.layout:"), std::string::npos);

  j = valid_json();
  j["dataset"]["last"] = -1;  // only the folder layout allows -1
  EXPECT_NE(problems(j).find("dataset.last:"), std::string::npos);

  // With a shared camera the calibration folder is not needed.
  fs::remove(base_ / "ds" / "calib");
  j = valid_json();
  EXPECT_NE(problems(j).find("dataset.calib_dir:"), std::string::npos);
  j["dataset"]["camera"] = {{"fx", 1000}};
  EXPECT_EQ(problems(j), "");
  j["dataset"]["camera"] = {{"exif", true}};
  EXPECT_EQ(problems(j), "");
  j["dataset"]["camera"] = {{"exif", true}, {"fx", 1000}};
  EXPECT_NE(problems(j).find("set either fx or exif"), std::string::npos);
  j["dataset"]["camera"] = {{"fy", 1000}};
  EXPECT_NE(problems(j).find("dataset.camera.fx: must be set when fy is"), std::string::npos);
}

TEST_F(ConfigValidate, RefineFocalNeedsTheFinalBa) {
  json j = valid_json();
  j["ba"]["refine_focal"] = true;
  EXPECT_EQ(problems(j), "");
  EXPECT_TRUE(parsed(j).ba.refine_focal);
  j["ba"]["final_max_iters"] = 0;
  EXPECT_NE(problems(j).find("ba.refine_focal: needs the final BA"), std::string::npos);
}

TEST_F(ConfigValidate, DistortionKeys) {
  json j = valid_json();
  j["dataset"]["distortion"] = {{"k1", -0.1}, {"k2", 0.01}};
  j["ba"]["refine_distortion"] = true;
  EXPECT_EQ(problems(j), "");
  const auto c = parsed(j);
  EXPECT_DOUBLE_EQ(c.dataset.k1, -0.1);
  EXPECT_DOUBLE_EQ(c.dataset.k2, 0.01);
  EXPECT_TRUE(c.ba.refine_distortion);
  const auto again = parse_config(sfm::config::to_json(c));  // config.json keeps them
  EXPECT_DOUBLE_EQ(again.dataset.k1, -0.1);
  EXPECT_TRUE(again.ba.refine_distortion);
  j["dataset"]["distortion"]["k1"] = 1.5;
  EXPECT_NE(problems(j).find("dataset.distortion.k1: must be in (-1, 1)"), std::string::npos);
  j["dataset"]["distortion"] = {{"p1", 0.0}};
  EXPECT_THROW(parsed(j), sfm::ConfigError);  // unknown key
}
