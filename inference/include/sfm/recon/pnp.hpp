#pragma once
// Camera pose from 2D-3D correspondences with known K.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "sfm/recon/geometry.hpp"

namespace sfm::recon {

// P3P (Grunert 1841, in the form of Haralick et al. 1994): the camera poses that map three world
// points onto three unit bearing vectors. The distances along the rays solve a quartic; each real
// root gives one pose (R, t from the three point pairs by absolute orientation). Up to 4 poses;
// empty for a degenerate (collinear) triple.
std::vector<Pose> p3p(const std::array<Eigen::Vector3d, 3>& X,
                      const std::array<Eigen::Vector3d, 3>& bearing);

// Rotation and translation with q_k = R p_k + t for k = 1..n (least squares, no scale; Kabsch).
Pose absolute_orientation(const std::vector<Eigen::Vector3d>& p,
                          const std::vector<Eigen::Vector3d>& q);

// Gauss-Newton on the pixel reprojection error over all given correspondences, pose update
// R <- exp(w) R, t <- t + dt. Returns the final RMS error in pixels.
double refine_pose(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                   const std::vector<Eigen::Vector2d>& uv, Pose& pose, int iterations = 10);

struct PnpParams {
  double threshold_px = 8.0;
  int min_inliers = 20;
  double min_inlier_ratio = 0.0;  // ok also needs inliers >= this share of the correspondences
  int max_iters = 1000;
  double confidence = 0.999;
  std::uint64_t seed = 0;
};

struct PnpResult {
  bool ok = false;
  Pose pose;
  std::vector<std::uint8_t> inlier;
  int inliers = 0;
  int iterations = 0;
};

// RANSAC over minimal P3P samples (each of the up to 4 poses scored; inlier: reprojection <
// threshold and in front of the camera), adaptive iterations N = log(1 - conf) / log(1 - w^3),
// then Gauss-Newton on the inliers and an inlier update (twice). ok = inliers >= min_inliers.
// (A 6-point DLT was tried first: on bunny's ~11 degree view of the object its 11-parameter
// projective fit is so ill-conditioned that the all-point DLT has a 42 px median error and no
// minimal sample reaches the 8 px threshold.)
PnpResult pnp_ransac(const Eigen::Matrix3d& K, const std::vector<Eigen::Vector3d>& X,
                     const std::vector<Eigen::Vector2d>& uv, const PnpParams& p);

}  // namespace sfm::recon
