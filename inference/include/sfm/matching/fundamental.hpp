#pragma once
// Fundamental matrix estimation: x2^T F x1 = 0 for corresponding pixels x1 (image 1), x2 (image 2).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sfm::matching {

using Points2 = std::vector<Eigen::Vector2d>;

// Hartley's normalisation: T maps the points to centroid 0 and mean distance sqrt(2).
// Throws sfm::GeometryError if all points coincide.
Eigen::Matrix3d hartley_normalisation(const Points2& x);

// Normalised 8-point algorithm (Hartley 1997), n >= 8 correspondences:
//   1. normalise both point sets (T1, T2)
//   2. each correspondence gives one row of A f = 0 (f = F row-major); f = right singular vector of
//      the smallest singular value
//   3. enforce rank 2: zero the smallest singular value of F
//   4. denormalise F = T2^T F_n T1, scale to ||F||_F = 1
// Throws sfm::GeometryError if n < 8, the sizes differ, or a point set is degenerate.
Eigen::Matrix3d eight_point(const Points2& x1, const Points2& x2);

// Squared Sampson distance (first-order geometric error, pixels^2):
//   (x2^T F x1)^2 / ((F x1)_0^2 + (F x1)_1^2 + (F^T x2)_0^2 + (F^T x2)_1^2)
double sampson_sq(const Eigen::Matrix3d& F, const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);

struct RansacParams {
  double threshold_px = 3.0;  // inlier: Sampson distance < threshold
  double confidence = 0.999;
  int max_iters = 2000;
  std::uint64_t seed = 0;
};

struct RansacResult {
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  std::vector<std::uint8_t> inlier;  // per correspondence
  int inliers = 0;
  int iterations = 0;
};

// RANSAC over minimal 8-point samples:
//   - score = inlier count, ties broken by the smaller sum of squared Sampson errors of inliers
//   - adaptive iteration count N = log(1 - confidence) / log(1 - w^8) (w = best inlier share),
//     capped at max_iters
//   - final F refitted with eight_point on all inliers, inliers recomputed (repeated while the
//     inlier set grows, at most 3 times)
// Deterministic for a seed. Fewer than 8 correspondences, or no non-degenerate sample, gives
// inliers = 0.
RansacResult ransac_fundamental(const Points2& x1, const Points2& x2, const RansacParams& p);

// The final step of ransac_fundamental, shared with the GPU RANSAC: inliers of F0 (Sampson <
// threshold); if >= 8, refit with eight_point on the inliers while the inlier set grows (at most 3
// times). iterations = 0. A zero F0 gives inliers = 0.
RansacResult refit_fundamental(const Points2& x1, const Points2& x2, const Eigen::Matrix3d& F0,
                               double threshold_px);

}  // namespace sfm::matching
