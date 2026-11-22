#include "sfm/eval/pose_error.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"

namespace sfm::eval {

namespace {

Eigen::Matrix3Xd centres(const std::vector<Pose>& poses) {
  Eigen::Matrix3Xd c(3, static_cast<Eigen::Index>(poses.size()));
  for (std::size_t i = 0; i < poses.size(); ++i) {
    c.col(static_cast<Eigen::Index>(i)) = poses[i].centre();
  }
  return c;
}

double max_of(const std::vector<double>& v) {
  return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

}  // namespace

double PoseErrors::max_rotation_deg() const { return max_of(rotation_deg); }
double PoseErrors::max_centre_pct() const { return max_of(centre_pct); }

PoseErrors pose_error(const std::vector<Pose>& estimated, const std::vector<Pose>& truth) {
  if (estimated.size() != truth.size()) {
    throw GeometryError("pose_error: " + std::to_string(estimated.size()) + " estimated vs " +
                        std::to_string(truth.size()) + " true cameras");
  }
  const Eigen::Matrix3Xd c_est = centres(estimated);
  const Eigen::Matrix3Xd c_true = centres(truth);

  PoseErrors out;
  out.alignment = umeyama(c_est, c_true);  // throws for < 3 cameras or collinear centres
  const Eigen::Vector3d centroid = c_true.rowwise().mean();
  out.scene_radius = (c_true.colwise() - centroid).colwise().norm().mean();
  if (!(out.scene_radius > 0.0)) throw GeometryError("pose_error: true centres coincide");

  constexpr double kRadToDeg = 57.295779513082320876;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    // X_true = s R_a X_est + t_a, so in the true frame camera i rotates by R_est * R_a^T.
    const Eigen::Matrix3d r_aligned = estimated[i].R * out.alignment.R.transpose();
    const Eigen::AngleAxisd diff(truth[i].R * r_aligned.transpose());
    out.rotation_deg.push_back(diff.angle() * kRadToDeg);
    const Eigen::Vector3d c = out.alignment.apply(c_est.col(static_cast<Eigen::Index>(i)));
    out.centre_pct.push_back(100.0 * (c - truth[i].centre()).norm() / out.scene_radius);
  }
  return out;
}

}  // namespace sfm::eval
