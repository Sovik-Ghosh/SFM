#pragma once
// Two- and multi-view geometry for calibrated cameras. Conventions as in sfm/eval/pose.hpp:
// x_cam = R X + t, pixels x ~ K x_cam. "Normalised" coordinates are K^-1 x (the camera ray at z=1).

#include <array>
#include <vector>

#include <Eigen/Core>

#include "sfm/eval/pose.hpp"

namespace sfm::recon {

using eval::Pose;
using Mat34 = Eigen::Matrix<double, 3, 4>;

// [v]_x: [v]_x w = v x w.
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

// Nearest essential matrix (Frobenius): singular values (s1, s2, s3) -> (1, 1, 0).
Eigen::Matrix3d nearest_essential(const Eigen::Matrix3d& M);

// E = K2^T F K1, projected onto the essential manifold. x2n^T E x1n = 0 for normalised points.
Eigen::Matrix3d essential_from_fundamental(const Eigen::Matrix3d& F, const Eigen::Matrix3d& K1,
                                           const Eigen::Matrix3d& K2);

// The four (R, t) with E ~ [t]_x R (||t|| = 1): (U W V^T, +-u3), (U W^T V^T, +-u3), det R = +1.
std::array<Pose, 4> decompose_essential(const Eigen::Matrix3d& E);

// Linear (DLT) triangulation from n >= 2 views: each view k gives two rows of A X = 0,
//   x_k (P_k row 2) - (P_k row 0),  y_k (P_k row 2) - (P_k row 1),
// and X is the right singular vector of the smallest singular value (homogeneous least squares).
// P are 3x4 cameras in the same coordinates as x (normalised [R | t] with normalised points, or
// K [R | t] with pixels). Returns false if the point is at infinity (w ~ 0) or n < 2.
bool triangulate_dlt(const std::vector<Mat34>& P, const std::vector<Eigen::Vector2d>& x,
                     Eigen::Vector3d& X);

// Of the four decompositions, the one with the most normalised correspondences triangulated in
// front of both cameras (camera 1 = [I | 0]). `in_front` receives that count, `mask` (optional,
// size n) which points passed.
Pose select_relative_pose(const Eigen::Matrix3d& E, const std::vector<Eigen::Vector2d>& x1n,
                          const std::vector<Eigen::Vector2d>& x2n, int& in_front,
                          std::vector<std::uint8_t>* mask = nullptr);

// Pixel projection and helpers.
Eigen::Vector2d project_px(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X);
double depth(const Pose& pose, const Eigen::Vector3d& X);  // z in the camera frame
double reprojection_error(const Eigen::Matrix3d& K, const Pose& pose, const Eigen::Vector3d& X,
                          const Eigen::Vector2d& uv);
// Angle (degrees) between the rays from camera centres c1 and c2 to X.
double ray_angle_deg(const Eigen::Vector3d& c1, const Eigen::Vector3d& c2,
                     const Eigen::Vector3d& X);

// Pixel -> normalised coordinates.
Eigen::Vector2d normalise(const Eigen::Matrix3d& K, const Eigen::Vector2d& uv);
Mat34 pose_matrix(const Pose& pose);  // [R | t]

// Rotation exp([w]_x) (Rodrigues) and its inverse; angle-axis vector w, |w| = angle in radians.
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w);
Eigen::Vector3d log_so3(const Eigen::Matrix3d& R);
// Nearest rotation (Frobenius) to M, det = +1.
Eigen::Matrix3d nearest_rotation(const Eigen::Matrix3d& M);

}  // namespace sfm::recon
