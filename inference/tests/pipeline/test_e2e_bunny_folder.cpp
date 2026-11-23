// The bunny as a user's photo folder: PNG images and masks with arbitrary names, calibration by
// file stem ("folder" layout). It must give exactly the result of the numbered NNNN.ppm layout,
// and the scene export must hold the images and a COLMAP model of the real image size.
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <stb_image.h>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& f) {
  std::ifstream in(f);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void to_png(const fs::path& from, const fs::path& to) {
  int w = 0;
  int h = 0;
  int n = 0;
  unsigned char* px = stbi_load(from.string().c_str(), &w, &h, &n, 0);
  ASSERT_NE(px, nullptr) << from;
  const int stride = w * n;
  ASSERT_NE(stbi_write_png(to.string().c_str(), w, h, n, px, stride), 0) << to;
  stbi_image_free(px);
}

// "aa_IMG", "ab_IMG", ...: no digits to rely on, and sorting keeps the bunny's order.
std::string stem(int i) {
  return std::string{static_cast<char>('a' + i / 26), static_cast<char>('a' + i % 26)} + "_IMG";
}

}  // namespace

TEST(E2eBunnyFolder, RenamedPngsGiveTheNumberedResult) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path work = fs::temp_directory_path() / "sfm_e2e_bunny_folder";
  fs::remove_all(work);
  const fs::path data = work / "photos";
  for (const char* d : {"images", "masks", "calib"}) fs::create_directories(data / d);
  for (int i = 0; i < 36; ++i) {
    char id[8];
    std::snprintf(id, sizeof id, "%04d", i);
    to_png(src / "data/bunny/images" / (std::string(id) + ".ppm"),
           data / "images" / (stem(i) + ".png"));
    to_png(src / "data/bunny/silhouettes" / (std::string(id) + ".pgm"),
           data / "masks" / (stem(i) + ".png"));
    fs::copy_file(src / "data/bunny/calib" / (std::string(id) + ".txt"),
                  data / "calib" / (stem(i) + ".txt"));
  }

  auto cfg = sfm::config::load_config(src / "configs" / "bunny_sift_tuned2.json", src);
  cfg.run.output_dir = work / "out";
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  cfg.run.name = "numbered";
  const auto numbered = sfm::pipeline::run_pipeline(cfg, {}, log);

  cfg.run.name = "folder";
  cfg.dataset.root = data;
  cfg.dataset.images_dir = "images";
  cfg.dataset.masks_dir = "masks";
  cfg.dataset.calib_dir = "calib";
  cfg.dataset.layout = "folder";
  cfg.dataset.first = 0;
  cfg.dataset.last = -1;
  cfg.output.colmap_scene = true;
  const auto folder = sfm::pipeline::run_pipeline(cfg, {}, log);
  EXPECT_EQ(read_file(folder.run_dir / "metrics.json"),
            read_file(numbered.run_dir / "metrics.json"));

  // Scene export: images copied under their own names, cameras at the true size 1024 x 768.
  const fs::path scene = folder.run_dir / "scene";
  EXPECT_TRUE(fs::exists(scene / "images" / "aa_IMG.png"));
  EXPECT_TRUE(fs::exists(scene / "images" / "bj_IMG.png"));  // image 35
  const std::string cams = read_file(scene / "sparse/0/cameras.txt");
  EXPECT_NE(cams.find("\n1 PINHOLE 1024 768 "), std::string::npos) << cams.substr(0, 200);
  EXPECT_NE(read_file(scene / "sparse/0/images.txt").find(" aa_IMG.png\n"), std::string::npos);
  // The plain colmap/ export names the source files.
  EXPECT_NE(read_file(folder.run_dir / "colmap/images.txt").find(" aa_IMG.png\n"),
            std::string::npos);
  fs::remove_all(work);
}
