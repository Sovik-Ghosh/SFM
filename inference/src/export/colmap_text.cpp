// COLMAP text format: https://colmap.github.io/format.html#text-format
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
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

std::ofstream open(const std::filesystem::path& file, bool binary) {
  std::ofstream out(file, binary ? std::ios::binary : std::ios::out);
  if (!out) throw IoError("write_colmap: cannot open " + file.string());
  return out;
}

// The model in COLMAP's terms, built once for both formats. Ids are 1-based; image i -> id i + 1,
// camera i -> id i + 1 (one camera per image), track t -> point id t + 1.
struct ColmapModel {
  struct Camera {
    int id = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;  // COLMAP pixel convention (+0.5)
  };
  recon::Radial dist;  // any(): OPENCV model, else PINHOLE
  struct Image {
    int id = 0;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    std::string name;
    std::vector<double> xy;           // 2 per keypoint, + 0.5
    std::vector<std::int64_t> point;  // point id per keypoint, -1 if none
  };
  struct Point {
    std::int64_t id = 0;
    Eigen::Vector3d X;
    Rgb colour{};
    double error = 0.0;
    std::vector<std::pair<int, int>> track;  // (image id, keypoint index)
  };
  int width = 0;
  int height = 0;
  std::vector<Camera> cameras;
  std::vector<Image> images;
  std::vector<Point> points;
};

ColmapModel build_model(const recon::Reconstruction& rec, const recon::TrackIndex& index,
                        const std::vector<features::Features>& features,
                        const std::vector<std::string>& names, int width, int height,
                        const std::vector<Rgb>& colours, const recon::Radial& dist) {
  ColmapModel m;
  m.width = width;
  m.height = height;
  m.dist = dist;

  // keypoint -> track, per image; keypoint -> observation used for its point (not an outlier).
  std::vector<std::vector<int>> track_of(features.size());
  std::vector<std::vector<std::uint8_t>> used_kp(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) {
    track_of[i].assign(features[i].size(), -1);
    used_kp[i].assign(features[i].size(), 0);
    for (const auto& [t, kp] : index.in_image(static_cast<int>(i))) {
      track_of[i][static_cast<std::size_t>(kp)] = t;
    }
  }
  for (int t = 0; t < index.track_count(); ++t) {
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      if (recon::observation_used(rec, index, o)) {
        used_kp[static_cast<std::size_t>(o->image)][static_cast<std::size_t>(o->keypoint)] = 1;
      }
    }
  }

  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    const auto& c = rec.cameras[i];
    if (!c.registered) continue;
    const int id = static_cast<int>(i) + 1;
    m.cameras.push_back({id, c.K(0, 0), c.K(1, 1), c.K(0, 2) + 0.5, c.K(1, 2) + 0.5});
    ColmapModel::Image img;
    img.id = id;
    img.q = Eigen::Quaterniond(c.pose.R);
    img.t = c.pose.t;
    img.name = i < names.size() ? names[i] : std::to_string(i);
    const auto& f = features[i];
    for (std::size_t k = 0; k < f.size(); ++k) {
      const int t = track_of[i][k];
      const bool has = t >= 0 && rec.has_point[static_cast<std::size_t>(t)] != 0 && used_kp[i][k];
      img.xy.push_back(f.xy[2 * k] + 0.5);
      img.xy.push_back(f.xy[2 * k + 1] + 0.5);
      img.point.push_back(has ? t + 1 : -1);
    }
    m.images.push_back(std::move(img));
  }

  for (int t = 0; t < index.track_count(); ++t) {
    const auto ti = static_cast<std::size_t>(t);
    if (rec.has_point[ti] == 0) continue;
    ColmapModel::Point p;
    p.id = t + 1;
    p.X = rec.points[ti];
    p.colour = ti < colours.size() ? colours[ti] : Rgb{128, 128, 128};
    int n = 0;
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      const auto& c = rec.cameras[static_cast<std::size_t>(o->image)];
      if (!recon::observation_used(rec, index, o)) continue;
      p.error += recon::reprojection_error(c.K, c.pose, p.X, o->uv);
      ++n;
      p.track.emplace_back(o->image + 1, o->keypoint);
    }
    if (n > 0) p.error /= n;
    m.points.push_back(std::move(p));
  }
  return m;
}

void write_text(const std::filesystem::path& dir, const ColmapModel& m) {
  std::filesystem::create_directories(dir);
  char buf[256];
  auto cams = open(dir / "cameras.txt", false);
  cams << "# Camera list with one line of data per camera:\n"
          "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
  for (const auto& c : m.cameras) {
    if (m.dist.any()) {
      std::snprintf(buf, sizeof buf, "%d OPENCV %d %d %.12g %.12g %.12g %.12g %.12g %.12g 0 0\n",
                    c.id, m.width, m.height, c.fx, c.fy, c.cx, c.cy, m.dist.k1, m.dist.k2);
    } else {
      std::snprintf(buf, sizeof buf, "%d PINHOLE %d %d %.12g %.12g %.12g %.12g\n", c.id, m.width,
                    m.height, c.fx, c.fy, c.cx, c.cy);
    }
    cams << buf;
  }
  auto imgs = open(dir / "images.txt", false);
  imgs << "# Image list with two lines of data per image:\n"
          "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
          "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
  for (const auto& im : m.images) {
    std::snprintf(buf, sizeof buf, "%d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %d ", im.id,
                  im.q.w(), im.q.x(), im.q.y(), im.q.z(), im.t.x(), im.t.y(), im.t.z(), im.id);
    imgs << buf << im.name << "\n";
    for (std::size_t k = 0; k < im.point.size(); ++k) {
      std::snprintf(buf, sizeof buf, "%s%.8g %.8g %lld", k ? " " : "", im.xy[2 * k],
                    im.xy[2 * k + 1], static_cast<long long>(im.point[k]));
      imgs << buf;
    }
    imgs << "\n";
  }
  auto pts = open(dir / "points3D.txt", false);
  pts << "# 3D point list with one line of data per point:\n"
         "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
  for (const auto& p : m.points) {
    const auto id = static_cast<long long>(p.id);
    const Rgb& c = p.colour;
    std::snprintf(buf, sizeof buf, "%lld %.12g %.12g %.12g %d %d %d %.6g", id, p.X.x(), p.X.y(),
                  p.X.z(), c[0], c[1], c[2], p.error);
    pts << buf;
    for (const auto& [img, kp] : p.track) pts << " " << img << " " << kp;
    pts << "\n";
  }
}

template <class T>
void put(std::ofstream& out, T v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof v);  // little-endian hosts (x86-64, ARM64)
}

void write_binary(const std::filesystem::path& dir, const ColmapModel& m) {
  std::filesystem::create_directories(dir);
  constexpr int kPinhole = 1;  // COLMAP's model ids
  constexpr int kOpenCv = 4;
  auto cams = open(dir / "cameras.bin", true);
  put<std::uint64_t>(cams, m.cameras.size());
  for (const auto& c : m.cameras) {
    put<std::int32_t>(cams, c.id);
    put<std::int32_t>(cams, m.dist.any() ? kOpenCv : kPinhole);
    put<std::uint64_t>(cams, static_cast<std::uint64_t>(m.width));
    put<std::uint64_t>(cams, static_cast<std::uint64_t>(m.height));
    for (const double v : {c.fx, c.fy, c.cx, c.cy}) put<double>(cams, v);
    if (m.dist.any()) {
      for (const double v : {m.dist.k1, m.dist.k2, 0.0, 0.0}) put<double>(cams, v);
    }
  }
  auto imgs = open(dir / "images.bin", true);
  put<std::uint64_t>(imgs, m.images.size());
  for (const auto& im : m.images) {
    put<std::int32_t>(imgs, im.id);
    for (const double v : {im.q.w(), im.q.x(), im.q.y(), im.q.z()}) put<double>(imgs, v);
    for (const double v : {im.t.x(), im.t.y(), im.t.z()}) put<double>(imgs, v);
    put<std::int32_t>(imgs, im.id);  // camera id
    // The name with its terminating '\0'.
    imgs.write(im.name.c_str(), static_cast<std::streamsize>(im.name.size() + 1));
    put<std::uint64_t>(imgs, im.point.size());
    for (std::size_t k = 0; k < im.point.size(); ++k) {
      put<double>(imgs, im.xy[2 * k]);
      put<double>(imgs, im.xy[2 * k + 1]);
      put<std::int64_t>(imgs, im.point[k]);
    }
  }
  auto pts = open(dir / "points3D.bin", true);
  put<std::uint64_t>(pts, m.points.size());
  for (const auto& p : m.points) {
    put<std::uint64_t>(pts, static_cast<std::uint64_t>(p.id));
    for (const double v : {p.X.x(), p.X.y(), p.X.z()}) put<double>(pts, v);
    for (const std::uint8_t v : p.colour) put<std::uint8_t>(pts, v);
    put<double>(pts, p.error);
    put<std::uint64_t>(pts, p.track.size());
    for (const auto& [img, kp] : p.track) {
      put<std::int32_t>(pts, img);
      put<std::int32_t>(pts, kp);
    }
  }
  if (!cams || !imgs || !pts) throw IoError("write_colmap: write failed in " + dir.string());
}

}  // namespace

void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names, int width, int height,
                       const std::vector<std::uint8_t>& gray) {
  write_colmap_text(dir, rec, index, features, names, width, height, gray_to_rgb(gray));
}

void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names, int width, int height,
                       const std::vector<Rgb>& colours) {
  write_text(dir, build_model(rec, index, features, names, width, height, colours, {}));
}

void write_colmap(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                  const recon::TrackIndex& index, const std::vector<features::Features>& features,
                  const std::vector<std::string>& names, int width, int height,
                  const std::vector<Rgb>& colours, const recon::Radial& dist) {
  const ColmapModel m = build_model(rec, index, features, names, width, height, colours, dist);
  write_text(dir, m);
  write_binary(dir, m);
}

void write_colmap_scene(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                        const recon::TrackIndex& index,
                        const std::vector<features::Features>& features,
                        const std::vector<std::filesystem::path>& images, int width, int height,
                        const std::vector<Rgb>& colours, const recon::Radial& dist) {
  namespace fs = std::filesystem;
  const fs::path out_images = dir / "images";
  fs::create_directories(out_images);
  std::vector<std::string> names;
  names.reserve(images.size());
  for (std::size_t i = 0; i < images.size(); ++i) {
    const fs::path& src = images[i];
    std::string ext = src.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (!dist.any() && (ext == ".jpg" || ext == ".jpeg" || ext == ".png")) {
      names.push_back(src.filename().string());
      std::error_code ec;
      fs::copy_file(src, out_images / names.back(), fs::copy_options::overwrite_existing, ec);
      if (ec) throw IoError("write_colmap_scene: copy " + src.string() + ": " + ec.message());
      continue;
    }
    names.push_back(src.stem().string() + ".png");  // PPM / PGM, or undistorted images
    int w = 0;
    int h = 0;
    int n = 0;
    unsigned char* px = stbi_load(src.string().c_str(), &w, &h, &n, 0);
    if (px == nullptr) {
      throw IoError("write_colmap_scene: read " + src.string() + ": " + stbi_failure_reason());
    }
    std::vector<unsigned char> buf(px, px + static_cast<std::size_t>(w) * h * n);
    stbi_image_free(px);
    if (dist.any() && i < rec.cameras.size()) {
      // Each output pixel (ideal, pinhole) takes the recorded pixel the lens put it at (bilinear).
      const Eigen::Matrix3d& K = rec.cameras[i].K;
      std::vector<unsigned char> und(buf.size(), 0);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const Eigen::Vector2d s = recon::distort(K, dist, Eigen::Vector2d(x, y));
          const int x0 = static_cast<int>(std::floor(s.x()));
          const int y0 = static_cast<int>(std::floor(s.y()));
          if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;  // outside: black
          const double fx = s.x() - x0;
          const double fy = s.y() - y0;
          for (int c = 0; c < n; ++c) {
            const auto at = [&](int xx, int yy) {
              return static_cast<double>(
                  buf[(static_cast<std::size_t>(yy) * w + static_cast<std::size_t>(xx)) * n + c]);
            };
            const double v = (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0)) +
                             fy * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
            und[(static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)) * n + c] =
                static_cast<unsigned char>(std::lround(std::clamp(v, 0.0, 255.0)));
          }
        }
      }
      buf.swap(und);
    }
    const int stride = w * n;
    const std::string out = (out_images / names.back()).string();
    const int ok = stbi_write_png(out.c_str(), w, h, n, buf.data(), stride);
    if (ok == 0) throw IoError("write_colmap_scene: cannot write " + names.back());
  }
  // The images are pinhole now: the undistorted keypoints, no distortion parameters.
  write_colmap(dir / "sparse" / "0", rec, index, features, names, width, height, colours);
}

}  // namespace sfm::exporter
