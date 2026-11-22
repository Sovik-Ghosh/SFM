#pragma once
// Stage contract of the reconstruction stage, and an index from tracks to pixel observations.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "sfm/data/rig.hpp"
#include "sfm/features/features.hpp"
#include "sfm/matching/matches.hpp"
#include "sfm/recon/geometry.hpp"

namespace sfm::recon {

struct Camera {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();  // fixed (from the calibration)
  Pose pose;
  bool registered = false;
};

struct Reconstruction {
  std::vector<Camera> cameras;          // one per image
  std::vector<Eigen::Vector3d> points;  // one per track; valid where has_point
  std::vector<std::uint8_t> has_point;  // one per track
  std::vector<int> order;               // registration order; order[0], order[1] = initial pair
  // Per TrackIndex observation (TrackIndex::index_of): 1 = rejected for its point (wrong match,
  // e.g. the same feature on another window of a repeated facade). Empty = none rejected.
  std::vector<std::uint8_t> obs_outlier;
  // Camera rig (P9b, reconstruction.rig): the images of one frame are registered together and keep
  // the rig's offsets (image pose = offset * frame pose). Null: every image is independent.
  std::shared_ptr<const data::Rig> rig;

  int registered_count() const;
  int point_count() const;
};

// Rig poses (sfm/data/rig.hpp): the frame pose of image i from its own pose, and the pose of image
// i from its frame's pose.
Pose rig_frame_pose(const data::Rig& rig, int image, const Pose& image_pose);
Pose rig_image_pose(const data::Rig& rig, int image, const Pose& frame_pose);

// One observation of a track: image, keypoint, pixel position.
struct TrackObs {
  int image = 0;
  int keypoint = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
};

// Tracks with their pixel positions, by track and by image.
class TrackIndex {
 public:
  TrackIndex(const matching::Tracks& tracks, const std::vector<features::Features>& features);

  int track_count() const { return static_cast<int>(offset_.size()) - 1; }
  int image_count() const { return static_cast<int>(by_image_.size()); }
  // Observations of track t, sorted by image.
  const TrackObs* begin(int t) const { return obs_.data() + offset_[static_cast<std::size_t>(t)]; }
  const TrackObs* end(int t) const {
    return obs_.data() + offset_[static_cast<std::size_t>(t) + 1];
  }
  std::size_t obs_count() const { return obs_.size(); }
  std::size_t index_of(const TrackObs* o) const {
    return static_cast<std::size_t>(o - obs_.data());
  }
  // (track, keypoint) of every tracked keypoint of image i, by keypoint.
  const std::vector<std::pair<int, int>>& in_image(int i) const {
    return by_image_[static_cast<std::size_t>(i)];
  }

 private:
  std::vector<TrackObs> obs_;
  std::vector<std::size_t> offset_;
  std::vector<std::vector<std::pair<int, int>>> by_image_;
};

// True if observation o of a track is used for its point: its camera is registered and it is not
// marked as an outlier.
bool observation_used(const Reconstruction& rec, const TrackIndex& index, const TrackObs* o);

// Every observation of a reconstructed point in a registered camera: (camera, track, uv).
struct PointObs {
  int camera = 0;
  int track = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
};
std::vector<PointObs> point_observations(const Reconstruction& rec, const TrackIndex& index);

// Reprojection error (pixels) of every point observation, in point_observations order.
std::vector<double> reprojection_errors(const Reconstruction& rec, const TrackIndex& index);

}  // namespace sfm::recon
