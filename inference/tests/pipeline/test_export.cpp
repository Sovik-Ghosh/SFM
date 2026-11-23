#include <gtest/gtest.h>

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
