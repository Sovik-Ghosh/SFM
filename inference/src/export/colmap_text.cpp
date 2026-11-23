// COLMAP text format: https://colmap.github.io/format.html#text-format
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>
#include <stb_image.h>

#include "sfm/core/error.hpp"
#include "sfm/export/export.hpp"

// PNG writing for the scene export; static, so it cannot clash with another stb_image_write copy.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace sfm::exporter {

namespace {

std::ofstream open(const std::filesystem::path& file) {
  std::ofstream out(file);
  if (!out) throw IoError("write_colmap_text: cannot open " + file.string());
  return out;
}

}  // namespace

void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names, int width, int height,
                       const std::vector<std::uint8_t>& gray) {
  std::filesystem::create_directories(dir);
  char buf[256];

  auto cams = open(dir / "cameras.txt");
  cams << "# Camera list with one line of data per camera:\n"
          "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    const auto& c = rec.cameras[i];
    if (!c.registered) continue;
    std::snprintf(buf, sizeof buf, "%zu PINHOLE %d %d %.12g %.12g %.12g %.12g\n", i + 1, width,
                  height, c.K(0, 0), c.K(1, 1), c.K(0, 2) + 0.5, c.K(1, 2) + 0.5);
    cams << buf;
  }

  // keypoint -> track, per image (for POINTS2D and the points3D track lists).
  std::vector<std::vector<int>> track_of(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    track_of[i].assign(features[i].size(), -1);
    for (const auto& [t, kp] : index.in_image(static_cast<int>(i))) {
      track_of[i][static_cast<std::size_t>(kp)] = t;
    }
  }

  // keypoint -> observation used for its point (not an outlier), per image.
  std::vector<std::vector<std::uint8_t>> used_kp(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) used_kp[i].assign(features[i].size(), 0);
  for (int t = 0; t < index.track_count(); ++t) {
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      if (recon::observation_used(rec, index, o)) {
        used_kp[static_cast<std::size_t>(o->image)][static_cast<std::size_t>(o->keypoint)] = 1;
      }
    }
  }

  auto imgs = open(dir / "images.txt");
  imgs << "# Image list with two lines of data per image:\n"
          "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
          "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    const auto& c = rec.cameras[i];
    if (!c.registered) continue;
    const Eigen::Quaterniond q(c.pose.R);
    std::snprintf(buf, sizeof buf, "%zu %.12g %.12g %.12g %.12g %.12g %.12g %.12g %zu ", i + 1,
                  q.w(), q.x(), q.y(), q.z(), c.pose.t.x(), c.pose.t.y(), c.pose.t.z(), i + 1);
    imgs << buf << (i < names.size() ? names[i] : std::to_string(i)) << "\n";
    const auto& f = features[i];
    for (std::size_t k = 0; k < f.size(); ++k) {
      const int t = track_of[i][k];
      const long id =
          (t >= 0 && rec.has_point[static_cast<std::size_t>(t)] != 0 && used_kp[i][k] != 0) ? t + 1L
                                                                                            : -1L;
      std::snprintf(buf, sizeof buf, "%s%.8g %.8g %ld", k ? " " : "", f.xy[2 * k] + 0.5,
                    f.xy[2 * k + 1] + 0.5, id);
      imgs << buf;
    }
    imgs << "\n";
  }

  auto pts = open(dir / "points3D.txt");
  pts << "# 3D point list with one line of data per point:\n"
         "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
  for (int t = 0; t < index.track_count(); ++t) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    const auto& X = rec.points[static_cast<std::size_t>(t)];
    double err = 0.0;
    int n = 0;
    std::string track;
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      const auto& c = rec.cameras[static_cast<std::size_t>(o->image)];
      if (!recon::observation_used(rec, index, o)) continue;
      err += recon::reprojection_error(c.K, c.pose, X, o->uv);
      ++n;
      track += " " + std::to_string(o->image + 1) + " " + std::to_string(o->keypoint);
    }
    const int g =
        static_cast<std::size_t>(t) < gray.size() ? gray[static_cast<std::size_t>(t)] : 128;
    std::snprintf(buf, sizeof buf, "%d %.12g %.12g %.12g %d %d %d %.6g", t + 1, X.x(), X.y(), X.z(),
                  g, g, g, n ? err / n : 0.0);
    pts << buf << track << "\n";
  }
}

void write_colmap_scene(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                        const recon::TrackIndex& index,
                        const std::vector<features::Features>& features,
                        const std::vector<std::filesystem::path>& images, int width, int height,
                        const std::vector<std::uint8_t>& gray) {
  namespace fs = std::filesystem;
  const fs::path out_images = dir / "images";
  fs::create_directories(out_images);
  std::vector<std::string> names;
  names.reserve(images.size());
  for (const fs::path& src : images) {
    std::string ext = src.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
      names.push_back(src.filename().string());
      std::error_code ec;
      fs::copy_file(src, out_images / names.back(), fs::copy_options::overwrite_existing, ec);
      if (ec) throw IoError("write_colmap_scene: copy " + src.string() + ": " + ec.message());
      continue;
    }
    names.push_back(src.stem().string() + ".png");  // PPM / PGM: few tools read them
    int w = 0;
    int h = 0;
    int n = 0;
    unsigned char* px = stbi_load(src.string().c_str(), &w, &h, &n, 0);
    if (px == nullptr) {
      throw IoError("write_colmap_scene: read " + src.string() + ": " + stbi_failure_reason());
    }
    const int stride = w * n;
    const std::string out = (out_images / names.back()).string();
    const int ok = stbi_write_png(out.c_str(), w, h, n, px, stride);
    stbi_image_free(px);
    if (ok == 0) throw IoError("write_colmap_scene: cannot write " + names.back());
  }
  write_colmap_text(dir / "sparse" / "0", rec, index, features, names, width, height, gray);
}

}  // namespace sfm::exporter
