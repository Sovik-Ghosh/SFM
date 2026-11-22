#include "sfm/eval/synthetic.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"
#include "sfm/eval/rng.hpp"

namespace sfm::eval {

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool ok, const std::string& field, const std::string& rule) {
  if (!ok) throw ConfigError("synthetic." + field + ": " + rule);
}

void validate(const SyntheticConfig& c) {
  require(c.cameras >= 1, "cameras", "must be >= 1");
  require(c.points >= 1, "points", "must be >= 1");
  require(c.min_elevation_deg >= 0.0 && c.min_elevation_deg <= c.max_elevation_deg,
          "min_elevation_deg", "must be in [0, max_elevation_deg]");
  require(c.max_elevation_deg < 90.0, "max_elevation_deg", "must be below 90");
  require(c.box_half > 0.0, "box_half", "must be > 0");
  require(c.box_half * std::sqrt(3.0) < c.camera_distance, "box_half",
          "box_half * sqrt(3) must be below camera_distance (points in front of every camera)");
  require(c.focal_px > 0.0, "focal_px", "must be > 0");
  require(c.width >= 1 && c.height >= 1, "width/height", "must be >= 1");
  require(c.noise_px >= 0.0, "noise_px", "must be >= 0");
  require(c.outlier_fraction >= 0.0 && c.outlier_fraction <= 1.0, "outlier_fraction",
          "must be in [0, 1]");
}

}  // namespace

Pose look_at_origin(const Eigen::Vector3d& centre) {
  const Eigen::Vector3d forward = -centre.normalized();                                // camera z
  const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();  // camera x
  const Eigen::Vector3d down = forward.cross(right);                                   // camera y
  Eigen::Matrix3d R;
  R.row(0) = right.transpose();
  R.row(1) = down.transpose();
  R.row(2) = forward.transpose();
  return Pose::from_centre(R, centre);
}

Eigen::Vector2d project(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X) {
  return (K * (pose.R * X + pose.t)).hnormalized();
}

SyntheticScene make_synthetic_scene(const SyntheticConfig& cfg) {
  validate(cfg);
  Rng rng(cfg.seed);  // draw order: cameras, points, noise, outliers (part of the format)
  SyntheticScene scene;
  scene.K << cfg.focal_px, 0.0, cfg.width / 2.0,  //
      0.0, cfg.focal_px, cfg.height / 2.0,        //
      0.0, 0.0, 1.0;

  // Cameras: evenly spread in azimuth with a little jitter, random elevation.
  for (int i = 0; i < cfg.cameras; ++i) {
    const double az = 2.0 * kPi * (i + rng.uniform(-0.25, 0.25)) / cfg.cameras;
    const double el = rng.uniform(cfg.min_elevation_deg, cfg.max_elevation_deg) * kPi / 180.0;
    const Eigen::Vector3d c(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
    scene.poses.push_back(look_at_origin(cfg.camera_distance * c));
  }

  scene.points.resize(3, cfg.points);
  for (int j = 0; j < cfg.points; ++j) {
    for (int k = 0; k < 3; ++k) scene.points(k, j) = rng.uniform(-cfg.box_half, cfg.box_half);
  }

  for (int i = 0; i < cfg.cameras; ++i) {
    for (int j = 0; j < cfg.points; ++j) {
      Observation o{i, j, project(scene.K, scene.poses[i], scene.points.col(j)), false};
      if (cfg.noise_px > 0.0) {
        o.uv.x() += rng.normal(0.0, cfg.noise_px);
        o.uv.y() += rng.normal(0.0, cfg.noise_px);
      }
      scene.observations.push_back(o);
    }
  }

  // Outliers: exactly round(fraction * N) observations, chosen by a partial Fisher-Yates shuffle.
  const std::size_t n = scene.observations.size();
  const auto k =
      static_cast<std::size_t>(std::llround(cfg.outlier_fraction * static_cast<double>(n)));
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  for (std::size_t m = 0; m < k; ++m) {
    std::swap(order[m], order[m + rng.index(n - m)]);
    Observation& o = scene.observations[order[m]];
    o.uv = {rng.uniform(0.0, cfg.width), rng.uniform(0.0, cfg.height)};
    o.outlier = true;
  }
  return scene;
}

}  // namespace sfm::eval
