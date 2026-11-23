#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "sfm/export/export.hpp"
#include "sfm/recon/incremental.hpp"

#include "scene.hpp"

namespace fs = std::filesystem;

namespace {

std::vector<std::string> data_lines(const fs::path& f) {
  std::ifstream in(f);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line[0] != '#') out.push_back(line);
  }
  return out;
}

template <class T>
T get(std::ifstream& in) {
  T v{};
  in.read(reinterpret_cast<char*>(&v), sizeof v);
  return v;
}

}  // namespace

TEST(Export, PlyAndColmapText) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 4;
  c.points = 30;
  c.box_half = 20.0;
  const auto in = sfm::recon::test::make_input(c);
  const sfm::recon::TrackIndex idx(in.tracks, in.features);
  sfm::recon::IncrementalParams ip;
  ip.init_min_inliers = 20;  // 30 points in the scene
  auto rec = sfm::recon::reconstruct_incremental(in.pairs, in.features, idx, in.K, ip, {}, nullptr);
  rec.has_point[4] = 0;  // one track without a point
  const fs::path dir = fs::temp_directory_path() / "sfm_export_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::vector<std::uint8_t> gray(30, 77);

  sfm::exporter::write_ply(dir / "m.ply", rec, gray);
  std::ifstream ply(dir / "m.ply");
  std::stringstream ss;
  ss << ply.rdbuf();
  const std::string text = ss.str();
  EXPECT_NE(text.find("element vertex 29\n"), std::string::npos);
  EXPECT_NE(text.find(" 77 77 77\n"), std::string::npos);

  sfm::exporter::write_colmap_text(dir / "colmap", rec, idx, in.features, {"a", "b", "c", "d"},
                                   1000, 700, gray);
  const auto cams = data_lines(dir / "colmap" / "cameras.txt");
  ASSERT_EQ(cams.size(), 4u);
  // The given image size (not 2 cx x 2 cy), and cx, cy + 0.5 for COLMAP's pixel convention.
  EXPECT_EQ(cams[0].rfind("1 PINHOLE 1000 700 1200 1200 512.5 384.5", 0), 0u) << cams[0];
  const auto imgs = data_lines(dir / "colmap" / "images.txt");
  ASSERT_EQ(imgs.size(), 8u);  // two lines per image
  EXPECT_NE(imgs[0].find(" 1 a"), std::string::npos);
  EXPECT_NE(imgs[1].find(" -1"), std::string::npos);  // keypoint 4: track without a point
  const auto pts = data_lines(dir / "colmap" / "points3D.txt");
  ASSERT_EQ(pts.size(), 29u);
  // Point 1 is seen by all four images: 4 (IMAGE_ID, POINT2D_IDX) pairs after the 8 fields.
  std::istringstream p1(pts[0]);
  int n = 0;
  for (std::string tok; p1 >> tok;) ++n;
  EXPECT_EQ(n, 8 + 2 * 4);
  fs::remove_all(dir);
}

// The binary model must hold what the text model says: read it back with COLMAP's layout.
TEST(Export, ColmapBinaryMatchesText) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 4;
  c.points = 30;
  c.box_half = 20.0;
  const auto in = sfm::recon::test::make_input(c);
  const sfm::recon::TrackIndex idx(in.tracks, in.features);
  sfm::recon::IncrementalParams ip;
  ip.init_min_inliers = 20;
  auto rec = sfm::recon::reconstruct_incremental(in.pairs, in.features, idx, in.K, ip, {}, nullptr);
  rec.has_point[4] = 0;
  const fs::path dir = fs::temp_directory_path() / "sfm_export_bin_test";
  fs::remove_all(dir);
  std::vector<sfm::exporter::Rgb> colours(30, sfm::exporter::Rgb{10, 20, 30});
  colours[0] = {200, 100, 50};
  // Image 2 is portrait: each camera carries its own size.
  std::vector<sfm::exporter::ImageSize> sizes(4, sfm::exporter::ImageSize{1000, 700});
  sizes[1] = {700, 1000};
  sfm::exporter::write_colmap(dir, rec, idx, in.features, {"a.jpg", "b", "c", "d"}, sizes, colours);
  const auto txt_cams = data_lines(dir / "cameras.txt");
  ASSERT_EQ(txt_cams.size(), 4u);
  EXPECT_EQ(txt_cams[1].rfind("2 PINHOLE 700 1000 ", 0), 0u) << txt_cams[1];

  std::ifstream cams(dir / "cameras.bin", std::ios::binary);
  ASSERT_EQ(get<std::uint64_t>(cams), 4u);
  EXPECT_EQ(get<std::int32_t>(cams), 1);  // camera id
  EXPECT_EQ(get<std::int32_t>(cams), 1);  // PINHOLE
  EXPECT_EQ(get<std::uint64_t>(cams), 1000u);
  EXPECT_EQ(get<std::uint64_t>(cams), 700u);
  EXPECT_DOUBLE_EQ(get<double>(cams), 1200.0);
  EXPECT_DOUBLE_EQ(get<double>(cams), 1200.0);
  EXPECT_DOUBLE_EQ(get<double>(cams), 512.5);
  EXPECT_DOUBLE_EQ(get<double>(cams), 384.5);

  const auto txt_imgs = data_lines(dir / "images.txt");
  std::ifstream imgs(dir / "images.bin", std::ios::binary);
  ASSERT_EQ(get<std::uint64_t>(imgs), 4u);
  EXPECT_EQ(get<std::int32_t>(imgs), 1);
  std::istringstream line0(txt_imgs[0]);
  int id = 0;
  line0 >> id;
  for (int k = 0; k < 7; ++k) {  // QW QX QY QZ TX TY TZ, as in the text (12 digits)
    double v = 0.0;
    line0 >> v;
    EXPECT_NEAR(get<double>(imgs), v, 1e-9);
  }
  EXPECT_EQ(get<std::int32_t>(imgs), 1);  // camera id
  std::string name;
  std::getline(imgs, name, '\0');
  EXPECT_EQ(name, "a.jpg");
  const auto n2d = get<std::uint64_t>(imgs);
  std::istringstream pts2d(txt_imgs[1]);
  std::uint64_t count = 0;
  for (double x = 0, y = 0; pts2d >> x >> y;) {
    long long pid = 0;
    pts2d >> pid;
    EXPECT_NEAR(get<double>(imgs), x, 1e-4);
    EXPECT_NEAR(get<double>(imgs), y, 1e-4);
    EXPECT_EQ(get<std::int64_t>(imgs), pid);
    ++count;
  }
  EXPECT_EQ(n2d, count);

  std::ifstream pts(dir / "points3D.bin", std::ios::binary);
  const auto txt_pts = data_lines(dir / "points3D.txt");
  ASSERT_EQ(get<std::uint64_t>(pts), txt_pts.size());
  EXPECT_EQ(get<std::uint64_t>(pts), 1u);  // point id
  for (int k = 0; k < 3; ++k) get<double>(pts);
  EXPECT_EQ(get<std::uint8_t>(pts), 200);
  EXPECT_EQ(get<std::uint8_t>(pts), 100);
  EXPECT_EQ(get<std::uint8_t>(pts), 50);
  EXPECT_NE(txt_pts[0].find(" 200 100 50 "), std::string::npos) << txt_pts[0];
  get<double>(pts);  // the error
  // Seen by all four images; the first is image 1.
  EXPECT_EQ(get<std::uint64_t>(pts), 4u);
  EXPECT_EQ(get<std::int32_t>(pts), 1);
  fs::remove_all(dir);
}

// Distortion: OPENCV model with k1, k2 in text and binary.
TEST(Export, ColmapOpenCvModelWithDistortion) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 4;
  c.points = 30;
  c.box_half = 20.0;
  const auto in = sfm::recon::test::make_input(c);
  const sfm::recon::TrackIndex idx(in.tracks, in.features);
  sfm::recon::IncrementalParams ip;
  ip.init_min_inliers = 20;
  auto rec = sfm::recon::reconstruct_incremental(in.pairs, in.features, idx, in.K, ip, {}, nullptr);
  const fs::path dir = fs::temp_directory_path() / "sfm_export_opencv_test";
  fs::remove_all(dir);
  const std::vector<sfm::exporter::ImageSize> sizes(4, sfm::exporter::ImageSize{1000, 700});
  sfm::exporter::write_colmap(dir, rec, idx, in.features, {"a", "b", "c", "d"}, sizes, {},
                              sfm::recon::Radial{-0.1, 0.02});
  const auto cams = data_lines(dir / "cameras.txt");
  EXPECT_EQ(cams[0], "1 OPENCV 1000 700 1200 1200 512.5 384.5 -0.1 0.02 0 0");
  std::ifstream bin(dir / "cameras.bin", std::ios::binary);
  get<std::uint64_t>(bin);
  get<std::int32_t>(bin);
  EXPECT_EQ(get<std::int32_t>(bin), 4);  // OPENCV
  get<std::uint64_t>(bin);
  get<std::uint64_t>(bin);
  for (int k = 0; k < 4; ++k) get<double>(bin);
  EXPECT_DOUBLE_EQ(get<double>(bin), -0.1);
  EXPECT_DOUBLE_EQ(get<double>(bin), 0.02);
  EXPECT_DOUBLE_EQ(get<double>(bin), 0.0);
  EXPECT_DOUBLE_EQ(get<double>(bin), 0.0);
  fs::remove_all(dir);
}
