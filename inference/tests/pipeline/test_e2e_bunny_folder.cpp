// The bunny as a user's photo folder: PNG images and masks with arbitrary names, calibration by
// file stem ("folder" layout). It must give exactly the result of the numbered NNNN.ppm layout,
// and the scene export must hold the images and a COLMAP model of the real image size.
#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <stb_image.h>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/krt.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/recon/distortion.hpp"

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

// ba.refine_focal: the bunny with every focal length 5 % too short. Without refinement the
// median rotation error doubles (0.17 -> 0.35 deg); the focal search recovers fx to ~0.2 % and the
// error to ~0.18 deg.
TEST(E2eBunnyFolder, RefineFocalRecoversAWrongFocalLength) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path work = fs::temp_directory_path() / "sfm_e2e_bunny_focal";
  fs::remove_all(work);
  fs::create_directories(work / "calib");
  for (int i = 0; i < 36; ++i) {
    char id[16];
    std::snprintf(id, sizeof id, "%04d.txt", i);
    const auto P0 = sfm::data::read_projection(src / "data/bunny/calib" / id);
    auto krt = sfm::data::decompose_projection(P0);
    krt.K(0, 0) *= 0.95;
    krt.K(1, 1) *= 0.95;
    krt.K(0, 1) *= 0.95;
    const sfm::data::Mat34 P = sfm::data::compose_projection(krt);
    std::ofstream out(work / "calib" / id);
    out.precision(17);
    out << "CONTOUR\n";
    for (int r = 0; r < 3; ++r) {
      out << P(r, 0) << " " << P(r, 1) << " " << P(r, 2) << " " << P(r, 3) << "\n";
    }
  }
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_sift_tuned2.json", src);
  cfg.dataset.calib_dir = work / "calib";  // absolute: replaces root / calib_dir
  cfg.run.output_dir = work / "out";
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kInfo, &log_text, false);
  const auto wrong = sfm::pipeline::run_pipeline(cfg, {}, log);
  cfg.ba.refine_focal = true;
  cfg.run.name = "refined";
  const auto refined = sfm::pipeline::run_pipeline(cfg, {}, log);
  if (!wrong.metrics || !refined.metrics) {
    GTEST_FAIL() << "no metrics";
  }
  const auto& mw = *wrong.metrics;
  const auto& mr = *refined.metrics;
  EXPECT_GT(mw.rotation_median_deg, 0.3);
  EXPECT_LT(mr.rotation_median_deg, 0.2);
  EXPECT_EQ(mr.registered_ids.size(), 36u);
  const std::string text = log_text.str();
  const auto at = text.find("event=recon.focal_refined scale=");
  ASSERT_NE(at, std::string::npos);
  const double scale = std::stod(text.substr(at + 32));
  EXPECT_NEAR(scale, 1.0 / 0.95, 0.01) << text.substr(at, 120);
  fs::remove_all(work);
}

// dataset.distortion: the bunny recorded through a barrel lens (k1 = -0.15, each view warped with
// its true K). Ignoring the distortion costs accuracy (0.27 deg median); with the known k1 the
// result is that of the undistorted images (0.17-0.18 deg). The COLMAP camera is OPENCV.
TEST(E2eBunnyFolder, KnownDistortionIsUndone) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path work = fs::temp_directory_path() / "sfm_e2e_bunny_distortion";
  fs::remove_all(work);
  for (const char* d : {"images", "silhouettes"}) fs::create_directories(work / d);
  const sfm::recon::Radial lens{-0.15, 0.0};
  for (int i = 0; i < 36; ++i) {
    char id[8];
    std::snprintf(id, sizeof id, "%04d", i);
    const std::string n(id);
    const fs::path bunny = src / "data/bunny";
    const auto P = sfm::data::read_projection(bunny / "calib" / (n + ".txt"));
    const Eigen::Matrix3d K = sfm::data::decompose_projection(P).K;
    const auto rgb = sfm::data::load_rgb(bunny / "images" / (n + ".ppm"));
    const auto sil = sfm::data::load_gray(bunny / "silhouettes" / (n + ".pgm"));
    const int w = rgb.width;
    const int h = rgb.height;
    std::vector<unsigned char> out(rgb.rgb.size(), 0);
    std::vector<unsigned char> mask(sil.pixels.size(), 255);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // Recorded pixel (x, y) shows the ideal pixel undistort(x, y): nearest sample.
        const Eigen::Vector2d s = sfm::recon::undistort(K, lens, Eigen::Vector2d(x, y));
        const int sx = static_cast<int>(std::lround(s.x()));
        const int sy = static_cast<int>(std::lround(s.y()));
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const std::size_t to = static_cast<std::size_t>(y) * w + x;
        const std::size_t from = static_cast<std::size_t>(sy) * w + sx;
        for (int c = 0; c < 3; ++c) out[3 * to + c] = rgb.rgb[3 * from + c];
        mask[to] = sil.pixels[from];
      }
    }
    const std::string size = std::to_string(w) + " " + std::to_string(h);
    std::ofstream img(work / "images" / (n + ".ppm"), std::ios::binary);
    img << "P6\n" << size << "\n255\n" << std::string(out.begin(), out.end());
    std::ofstream msk(work / "silhouettes" / (n + ".pgm"), std::ios::binary);
    msk << "P5\n" << size << "\n255\n" << std::string(mask.begin(), mask.end());
  }
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_sift_tuned2.json", src);
  cfg.dataset.root = work;
  cfg.dataset.calib_dir = src / "data/bunny/calib";
  cfg.run.output_dir = work / "out";
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  cfg.run.name = "ignored";
  const auto ignored = sfm::pipeline::run_pipeline(cfg, {}, log);
  cfg.run.name = "known";
  cfg.dataset.k1 = lens.k1;
  const auto known = sfm::pipeline::run_pipeline(cfg, {}, log);
  if (!ignored.metrics || !known.metrics || !known.model) {
    GTEST_FAIL() << "no metrics";
  }
  EXPECT_GT(ignored.metrics->rotation_median_deg, 0.24);
  EXPECT_LT(known.metrics->rotation_median_deg, 0.2);
  EXPECT_EQ(known.metrics->registered_ids.size(), 36u);
  EXPECT_DOUBLE_EQ(known.model->k1, -0.15);
  std::ifstream cams(known.run_dir / "colmap" / "cameras.txt");
  std::stringstream text;
  text << cams.rdbuf();
  EXPECT_NE(text.str().find(" OPENCV 1024 768 "), std::string::npos);
  fs::remove_all(work);
}

// Portrait and landscape mixed: every third bunny view turned 90 degrees clockwise (768 x 1024),
// its calibration turned with it (u' = H - 1 - v, v' = u). Same accuracy as all-landscape, each
// COLMAP camera with its own size. memory.max_images = 4 with last = -1: the cap follows the
// folder.
TEST(E2eBunnyFolder, PortraitAndLandscapeMixed) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path work = fs::temp_directory_path() / "sfm_e2e_bunny_portrait";
  fs::remove_all(work);
  for (const char* d : {"images", "masks", "calib"}) fs::create_directories(work / d);
  for (int i = 0; i < 36; ++i) {
    char id[8];
    std::snprintf(id, sizeof id, "%04d", i);
    const std::string n(id);
    const fs::path bunny = src / "data/bunny";
    auto P = sfm::data::read_projection(bunny / "calib" / (n + ".txt"));
    const auto rgb = sfm::data::load_rgb(bunny / "images" / (n + ".ppm"));
    const auto sil = sfm::data::load_gray(bunny / "silhouettes" / (n + ".pgm"));
    int w = rgb.width;
    int h = rgb.height;
    std::vector<unsigned char> img = rgb.rgb;
    std::vector<unsigned char> msk = sil.pixels;
    if (i % 3 == 0) {  // rotate 90 degrees clockwise: (x, y) -> (h - 1 - y, x)
      std::vector<unsigned char> ri(img.size());
      std::vector<unsigned char> rm(msk.size());
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::size_t to = static_cast<std::size_t>(x) * h + (h - 1 - y);
          const std::size_t from = static_cast<std::size_t>(y) * w + x;
          for (int c = 0; c < 3; ++c) ri[3 * to + c] = img[3 * from + c];
          rm[to] = msk[from];
        }
      }
      Eigen::Matrix3d A;
      A << 0, -1, h - 1, 1, 0, 0, 0, 0, 1;
      P = A * P;
      img.swap(ri);
      msk.swap(rm);
      std::swap(w, h);
    }
    const std::string size = std::to_string(w) + " " + std::to_string(h);
    std::ofstream im(work / "images" / ("view_" + n + ".ppm"), std::ios::binary);
    im << "P6\n" << size << "\n255\n" << std::string(img.begin(), img.end());
    std::ofstream mk(work / "masks" / ("view_" + n + ".pgm"), std::ios::binary);
    mk << "P5\n" << size << "\n255\n" << std::string(msk.begin(), msk.end());
    std::ofstream cal(work / "calib" / ("view_" + n + ".txt"));
    cal.precision(17);
    cal << "CONTOUR\n";
    for (int r = 0; r < 3; ++r) {
      cal << P(r, 0) << " " << P(r, 1) << " " << P(r, 2) << " " << P(r, 3) << "\n";
    }
  }
  auto cfg = sfm::config::load_config(src / "configs" / "bunny_sift_tuned2.json", src);
  cfg.run.output_dir = work / "out";
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.dataset.root = work;
  cfg.dataset.images_dir = "images";
  cfg.dataset.masks_dir = "masks";
  cfg.dataset.calib_dir = "calib";
  cfg.dataset.layout = "folder";
  cfg.dataset.first = 0;
  cfg.dataset.last = -1;
  cfg.memory.max_images = 4;
  std::ostringstream log_text;
  sfm::core::Logger log(sfm::core::LogLevel::kWarn, &log_text, false);
  const auto r = sfm::pipeline::run_pipeline(cfg, {}, log);
  if (!r.metrics || !r.model) {
    GTEST_FAIL() << "no metrics";
  }
  EXPECT_EQ(r.metrics->registered_ids.size(), 36u);
  EXPECT_LT(r.metrics->rotation_median_deg, 0.25);  // landscape-only: 0.171 deg
  EXPECT_EQ(r.model->sizes[0], (std::array<int, 2>{768, 1024}));
  EXPECT_EQ(r.model->sizes[1], (std::array<int, 2>{1024, 768}));
  std::ifstream cams(r.run_dir / "colmap" / "cameras.txt");
  std::stringstream text;
  text << cams.rdbuf();
  EXPECT_NE(text.str().find("\n1 PINHOLE 768 1024 "), std::string::npos);
  EXPECT_NE(text.str().find("\n2 PINHOLE 1024 768 "), std::string::npos);
  fs::remove_all(work);
}
