#pragma once
// Stage contracts of the matching stage.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sfm::matching {

// Putative match of query keypoint `query` (image A) to train keypoint `train` (image B).
struct Match {
  int query = 0;
  int train = 0;
  int distance = 0;  // Hamming distance of the descriptors
};

// A verified image pair i < j: the F-RANSAC inliers and the fundamental matrix
// (x_j^T F x_i = 0, pixels).
struct PairMatches {
  int i = 0;
  int j = 0;
  std::vector<std::int32_t> idx;  // (m, 2): keypoint in image i, keypoint in image j
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();

  std::size_t size() const { return idx.size() / 2; }
};

// Feature tracks: each track is one 3D point seen in several images.
struct Tracks {
  std::vector<std::int32_t> obs;  // (O, 3): track id, image index, keypoint index;
                                  // sorted by track, then image; ids 0..count-1
  int count = 0;

  std::size_t size() const { return obs.size() / 3; }
};

}  // namespace sfm::matching
