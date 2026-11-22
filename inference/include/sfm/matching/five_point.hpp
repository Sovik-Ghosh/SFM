#pragma once
// Essential matrix from five calibrated correspondences (Nistér 2004, in the action-matrix form of
// Stewénius, Engels and Nistér 2006), and RANSAC over it. Unlike the 8-point F, the 5-point solver
// is not degenerate when the points lie on a plane (measured on ETH3D courtyard, mostly facades:
// with 8-point F verification 118 of 241 accepted pairs were mostly wrong).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sfm::matching {

// On a plane, two essential matrices (the "planar twin") explain the points equally well; RANSAC's
// inlier set is right either way, the relative pose must be chosen by cheirality / refinement.
//
// Up to 10 essential matrices with x2n^T E x1n = 0 for the five normalised correspondences
// (x = K^-1 pixel). Each has unit Frobenius norm. Empty for a degenerate sample.
std::vector<Eigen::Matrix3d> five_point(const Eigen::Vector2d x1n[5], const Eigen::Vector2d x2n[5]);

struct EssentialRansacParams {
  double threshold_px = 3.0;  // Sampson distance threshold, in pixels (converted with `focal`)
  double focal = 1.0;         // pixels per normalised unit (mean of fx of the two cameras)
  double confidence = 0.999;
  int max_iters = 1000;
  std::uint64_t seed = 0;
};

struct EssentialRansacResult {
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  std::vector<std::uint8_t> inlier;
  int inliers = 0;
  int iterations = 0;
};

// MSAC over five-point samples (score = sum of min(Sampson^2, thr^2) in normalised units scaled
// by focal^2), adaptive iterations N = log(1 - conf) / log(1 - w^5). Deterministic for a seed.
EssentialRansacResult ransac_essential(const std::vector<Eigen::Vector2d>& x1n,
                                       const std::vector<Eigen::Vector2d>& x2n,
                                       const EssentialRansacParams& p);

}  // namespace sfm::matching
