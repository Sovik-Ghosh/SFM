// COLMAP text format: https://colmap.github.io/format.html#text-format
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"
#include "sfm/export/export.hpp"

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
                       const std::vector<std::string>& names,
                       const std::vector<std::uint8_t>& gray) {
  std::filesystem::create_directories(dir);
  char buf[256];

  auto cams = open(dir / "cameras.txt");
  cams << "# Camera list with one line of data per camera:\n"
          "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
  // WIDTH / HEIGHT: the principal point is near the image centre on bunny; 2 cx x 2 cy is used
  // only as size metadata.
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    const auto& c = rec.cameras[i];
    if (!c.registered) continue;
    std::snprintf(buf, sizeof buf, "%zu PINHOLE %d %d %.12g %.12g %.12g %.12g\n", i + 1,
                  static_cast<int>(std::lround(2 * c.K(0, 2))),
                  static_cast<int>(std::lround(2 * c.K(1, 2))), c.K(0, 0), c.K(1, 1), c.K(0, 2),
                  c.K(1, 2));
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
      std::snprintf(buf, sizeof buf, "%s%.6g %.6g %ld", k ? " " : "", f.xy[2 * k], f.xy[2 * k + 1],
                    id);
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

}  // namespace sfm::exporter
