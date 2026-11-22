#pragma once
// Progress of a pipeline run, for live observers (the viewer). The sink is called on the thread
// that runs run_pipeline; it receives copies (snapshots), so it may hand them to another thread.
// A sink never changes the result: a run with a sink writes the same metrics.json as without.

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "sfm/ba/ba.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::pipeline {

enum class Stage : std::uint8_t;
struct Metrics;
struct Timings;

// The reconstruction as it stands, in dataset order.
struct SceneSnapshot {
  int sequence = 0;                // 0, 1, 2, ... per run (snapshots in order)
  std::vector<int> ids;            // dataset id per image
  std::vector<Eigen::Matrix3d> K;  // per image
  std::vector<Eigen::Matrix3d> R;  // per image (valid where registered)
  std::vector<Eigen::Vector3d> t;
  std::vector<std::uint8_t> registered;  // per image
  std::vector<int> order;                // registration order (image indices)
  std::vector<Eigen::Vector3d> points;   // reconstructed points only
  std::vector<std::uint8_t> point_gray;  // per point (pixel under its first observation)
  std::vector<float> point_error_px;     // per point: mean reprojection error over registered views
  std::vector<int> point_track_length;   // per point: observations in registered views
};

// Snapshot of a reconstruction (point_gray: per track, see track_gray). Used by the pipeline for
// its sink and by viewers that load a finished run.
SceneSnapshot make_snapshot(const recon::Reconstruction& rec, const recon::TrackIndex& index,
                            const std::vector<int>& ids, const std::vector<std::uint8_t>& gray,
                            int sequence);

// Gray value per track: the pixel under its first observation (images in track image order).
std::vector<std::uint8_t> track_gray(const recon::TrackIndex& index,
                                     const std::vector<data::Image8>& gray);

class ProgressSink {
 public:
  virtual ~ProgressSink() = default;
  virtual void stage_begin(Stage /*stage*/) {}
  virtual void stage_end(Stage /*stage*/, double /*seconds*/) {}
  // After the initial pair and after each registered camera.
  virtual void camera_registered(const SceneSnapshot& /*s*/) {}
  // After each bundle adjustment (and the point filtering / re-triangulation that follows it).
  virtual void ba_done(const SceneSnapshot& /*s*/, const ba::Report& /*r*/) {}
  virtual void finished(const Metrics& /*m*/, const Timings& /*t*/) {}
  // Polled between stages, images and registrations; true stops the run with sfm::Cancelled.
  virtual bool cancelled() const { return false; }
};

}  // namespace sfm::pipeline
