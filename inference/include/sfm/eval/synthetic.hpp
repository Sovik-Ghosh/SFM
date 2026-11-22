#pragma once
// Synthetic scenes with exact ground truth: cameras on a hemisphere looking at the origin (like the
// bunny rig), points uniform in a box around the origin, every point projected into every camera,
// optional Gaussian pixel noise and a fixed fraction of outlier observations. Same seed, same
// scene, on every compiler (see rng.hpp).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "sfm/eval/pose.hpp"

namespace sfm::eval {

struct SyntheticConfig {
  int cameras = 12;
  int points = 400;
  double camera_distance = 60.0;    // every camera centre is this far from the origin
  double min_elevation_deg = 20.0;  // above the z = 0 plane; must stay below 90
  double max_elevation_deg = 70.0;
  double box_half = 8.0;  // points uniform in [-box_half, box_half]^3
  double focal_px = 1200.0;
  int width = 1024;  // principal point at the image centre
  int height = 768;
  double noise_px = 0.0;          // sigma of Gaussian noise added to each image coordinate
  double outlier_fraction = 0.0;  // this share of observations gets a uniform random pixel instead
  std::uint64_t seed = 1;
};

struct Observation {
  int camera = 0;
  int point = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  bool outlier = false;
};

struct SyntheticScene {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();  // shared by all cameras
  std::vector<Pose> poses;
  Eigen::Matrix3Xd points;                // 3 x points
  std::vector<Observation> observations;  // camera-major: index = camera * points + point
};

// Throws sfm::ConfigError naming the field when the config is invalid (counts < 1, elevation
// outside [0, 90), noise < 0, fraction outside [0, 1], or a box so large that a point could lie
// behind a camera: box_half * sqrt(3) must be below camera_distance).
SyntheticScene make_synthetic_scene(const SyntheticConfig& cfg);

// Camera at `centre` looking at the origin, world +z up in the image. centre must not lie on the z
// axis (the image "up" direction would be undefined).
Pose look_at_origin(const Eigen::Vector3d& centre);

// Pinhole projection to pixels (no distortion). The point must be in front of the camera.
Eigen::Vector2d project(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X);

}  // namespace sfm::eval
