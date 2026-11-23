#include "sfm/viewer/scene.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include "sfm/viewer/colormap_hd.hpp"

namespace sfm::viewer {

namespace {

Vertex vtx(const Eigen::Vector3f& p, Rgb c, std::uint8_t a = 255) {
  return {p.x(), p.y(), p.z(), c[0], c[1], c[2], a};
}

void line(std::vector<Vertex>& out, const Eigen::Vector3f& a, const Eigen::Vector3f& b, Rgb c) {
  out.push_back(vtx(a, c));
  out.push_back(vtx(b, c));
}

void frustum(std::vector<Vertex>& out, const Eigen::Matrix3d& K, const eval::Pose& pose, int w,
             int h, double depth, Rgb c) {
  const auto corners = frustum_corners(K, pose, w, h, depth);
  const Eigen::Vector3f apex = pose.centre().cast<float>();
  for (std::size_t k = 0; k < 4; ++k) {
    line(out, apex, corners[k].cast<float>(), c);
    line(out, corners[k].cast<float>(), corners[(k + 1) % 4].cast<float>(), c);
  }
  // A tick on the top edge marks the image's "up" side.
  const Eigen::Vector3f top = (0.5 * (corners[0] + corners[1])).cast<float>();
  const Eigen::Vector3f up = (pose.R.transpose() * Eigen::Vector3d(0, -1, 0)).cast<float>();
  line(out, top, top + up * static_cast<float>(0.2 * depth), c);
}

// Frustum size: the configured image size, or the camera's own from its principal point.
int frustum_width(const SceneOptions& o, const Eigen::Matrix3d& K) {
  return o.image_width > 0 ? o.image_width : static_cast<int>(std::lround(2.0 * K(0, 2) + 1.0));
}
int frustum_height(const SceneOptions& o, const Eigen::Matrix3d& K) {
  return o.image_height > 0 ? o.image_height : static_cast<int>(std::lround(2.0 * K(1, 2) + 1.0));
}

}  // namespace

std::array<Eigen::Vector3d, 4> frustum_corners(const Eigen::Matrix3d& K, const eval::Pose& pose,
                                               int width, int height, double depth) {
  const Eigen::Matrix3d Kinv = K.inverse();
  const std::array<Eigen::Vector2d, 4> px = {Eigen::Vector2d(0, 0), Eigen::Vector2d(width, 0),
                                             Eigen::Vector2d(width, height),
                                             Eigen::Vector2d(0, height)};
  std::array<Eigen::Vector3d, 4> out;
  for (std::size_t k = 0; k < 4; ++k) {
    const Eigen::Vector3d xc = depth * (Kinv * px[k].homogeneous());
    out[k] = pose.R.transpose() * (xc - pose.t);
  }
  return out;
}

std::vector<Vertex> grid_lines(const Eigen::Vector3f& center, float radius,
                               const Eigen::Vector3f& up, int cells) {
  const Eigen::Vector3f n = up.normalized();
  const Eigen::Vector3f a =
      std::fabs(n.x()) < 0.9F ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitY();
  const Eigen::Vector3f e1 = (a - a.dot(n) * n).normalized();
  const Eigen::Vector3f e2 = n.cross(e1);
  const Eigen::Vector3f o = center - radius * n;
  const float step = 2.0F * radius / static_cast<float>(cells);
  const Rgb c{70, 70, 78};
  std::vector<Vertex> out;
  out.reserve(8 * static_cast<std::size_t>(cells) + 4);
  for (int k = -cells; k <= cells; ++k) {
    const float f = static_cast<float>(k) * step;
    const float ext = static_cast<float>(cells) * step;
    line(out, o + f * e1 - ext * e2, o + f * e1 + ext * e2, c);
    line(out, o + f * e2 - ext * e1, o + f * e2 + ext * e1, c);
  }
  return out;
}

Rgb camera_group_color(int group) {
  static constexpr Rgb kPalette[] = {{96, 165, 250},  {52, 211, 153},  {250, 204, 21},
                                     {244, 114, 182}, {167, 139, 250}, {34, 211, 238},
                                     {251, 146, 60},  {226, 232, 240}};
  return kPalette[static_cast<std::size_t>(((group % 8) + 8) % 8)];
}

Rgb camera_color(int k, int n) {
  return viridis(n > 1 ? static_cast<float>(k) / static_cast<float>(n - 1) : 1.0F);
}

SceneGeometry build_scene(const pipeline::SceneSnapshot& s, const SceneOptions& o) {
  SceneGeometry g;
  const std::size_t np = s.points.size();

  // Centre and robust radius of the points (cameras if there are no points yet).
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(np);
  for (const auto& p : s.points) pts.push_back(p.cast<float>());
  if (pts.empty()) {
    for (std::size_t i = 0; i < s.registered.size(); ++i) {
      if (s.registered[i] != 0) pts.push_back(eval::Pose{s.R[i], s.t[i]}.centre().cast<float>());
    }
  }
  if (!pts.empty()) {
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : pts) c += p;
    g.center = c / static_cast<float>(pts.size());
    std::vector<float> d;
    d.reserve(pts.size());
    for (const auto& p : pts) d.push_back((p - g.center).norm());
    g.radius = std::max(robust_range(d, 0.0F, 0.95F)[1], 1e-6F);
  }

  // World up: the axis of the registered cameras' "up" (-y image axis) vectors, signed by majority.
  // The principal axis of sum(u u^T) rather than the mean of u: some captures store images
  // rotated 180 degrees (Middlebury's ring datasets), whose ups cancel in a mean; when all
  // cameras agree both give the same direction.
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  std::vector<Eigen::Vector3d> ups;
  for (std::size_t i = 0; i < s.registered.size(); ++i) {
    if (s.registered[i] == 0) continue;
    ups.push_back(s.R[i].transpose() * Eigen::Vector3d(0, -1, 0));
    M += ups.back() * ups.back().transpose();
  }
  if (!ups.empty()) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
    Eigen::Vector3d axis = es.eigenvectors().col(2);  // largest eigenvalue
    int votes = 0;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& u : ups) {
      votes += u.dot(axis) >= 0.0 ? 1 : -1;
      mean += u;
    }
    if (votes < 0 || (votes == 0 && mean.dot(axis) < 0.0)) axis = -axis;
    g.up = axis.normalized().cast<float>();
  }

  // Point colours.
  std::vector<float> value(np, 0.0F);
  const int mode = static_cast<int>(o.color);
  for (std::size_t k = 0; k < np; ++k) {
    const Eigen::Vector3f p = s.points[k].cast<float>();
    value[k] = hd::point_value(mode, p.x(), p.y(), p.z(), g.up.x(), g.up.y(), g.up.z(),
                               s.point_gray[k], s.point_error_px[k], s.point_track_length[k]);
  }
  g.color_range =
      o.color == ColorMode::kGray ? std::array<float, 2>{0.0F, 1.0F} : robust_range(value);
  g.points.reserve(np);
  for (std::size_t k = 0; k < np; ++k) {
    Rgb c;
    if (o.color == ColorMode::kGray) {
      c = {s.point_gray[k], s.point_gray[k], s.point_gray[k]};
    } else {
      c = viridis(normalise(value[k], g.color_range[0], g.color_range[1]));
    }
    g.points.push_back(vtx(s.points[k].cast<float>(), c));
  }

  // Cameras.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  g.camera_centres.assign(s.registered.size(), Eigen::Vector3f(nan, nan, nan));
  const double depth = o.frustum_size * g.radius;
  for (std::size_t k = 0; k < s.order.size(); ++k) {
    const auto i = static_cast<std::size_t>(s.order[k]);
    const eval::Pose pose{s.R[i], s.t[i]};
    g.camera_centres[i] = pose.centre().cast<float>();
    g.camera_radius = std::max(g.camera_radius, (g.camera_centres[i] - g.center).norm());
    if (!o.frustums) continue;
    const bool sel = static_cast<int>(i) == o.selected_camera;
    const bool grouped = o.camera_group.size() == s.registered.size();
    const Rgb c = sel       ? Rgb{255, 80, 40}
                  : grouped ? camera_group_color(o.camera_group[i])
                            : camera_color(static_cast<int>(k), static_cast<int>(s.order.size()));
    frustum(g.lines, s.K[i], pose, frustum_width(o, s.K[i]), frustum_height(o, s.K[i]),
            sel ? 1.5 * depth : depth, c);
  }
  if (!o.ground_truth.empty()) {
    for (std::size_t i = 0; i < o.ground_truth.size() && i < s.registered.size(); ++i) {
      if (s.registered[i] == 0) continue;
      // Orange, as the Ground truth layer's swatch (theme kWarn 0xFFA552).
      frustum(g.lines, s.K[i], o.ground_truth[i], frustum_width(o, s.K[i]),
              frustum_height(o, s.K[i]), depth, Rgb{255, 165, 82});
      line(g.lines, o.ground_truth[i].centre().cast<float>(), g.camera_centres[i],
           Rgb{230, 60, 60});
    }
  }
  return g;
}

}  // namespace sfm::viewer
