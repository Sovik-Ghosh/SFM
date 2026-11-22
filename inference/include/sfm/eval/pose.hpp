#pragma once
// Camera pose (extrinsics): x_cam = R * X_world + t. R rotates world -> camera, det R = +1.
// Camera axes: x right, y down, z forward (the usual image convention).

#include <Eigen/Core>

namespace sfm::eval {

struct Pose {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  Eigen::Vector3d centre() const { return -R.transpose() * t; }  // camera centre, world frame

  static Pose from_centre(const Eigen::Matrix3d& R, const Eigen::Vector3d& centre) {
    return {R, -R * centre};
  }
};

}  // namespace sfm::eval
