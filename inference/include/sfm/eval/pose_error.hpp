#pragma once
// Pose accuracy of a reconstruction against ground truth. SfM recovers cameras only up to a
// similarity (frame, scale), so the estimated camera centres are first aligned to the true ones
// with Umeyama; then each camera gets
//   rotation error: angle of R_true * (R_est * R_align^T)^T, in degrees
//   centre error:   || align(C_est) - C_true || as % of the scene radius
// scene radius = mean distance of the true centres from their centroid.

#include <vector>

#include "sfm/eval/pose.hpp"
#include "sfm/eval/umeyama.hpp"

namespace sfm::eval {

struct PoseErrors {
  Similarity alignment;  // maps the estimated frame onto the true one
  double scene_radius = 0.0;
  std::vector<double> rotation_deg;  // per camera
  std::vector<double> centre_pct;    // per camera

  double max_rotation_deg() const;
  double max_centre_pct() const;
};

// Throws sfm::GeometryError if the counts differ, there are fewer than 3 cameras, or the true or
// estimated centres are collinear (the alignment is then not determined).
PoseErrors pose_error(const std::vector<Pose>& estimated, const std::vector<Pose>& truth);

}  // namespace sfm::eval
