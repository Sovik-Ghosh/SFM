#pragma once
// Ground truth in the viewer: the calibrated poses mapped into the reconstruction's frame (to draw
// them over the estimate) and the per-camera errors (the definitions of sfm::eval::pose_error).

#include <vector>

#include "sfm/eval/pose.hpp"
#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer {

struct GroundTruthView {
  std::vector<eval::Pose>
      aligned;  // per image, in the reconstruction frame (identity if unregistered)
  std::vector<double> rotation_deg;  // per image; -1 where not registered
  std::vector<double> centre_pct;    // per image; -1 where not registered
  bool valid = false;                // false with < 3 registered cameras
};

// truth: calibrated poses per image (same order as the snapshot).
GroundTruthView align_ground_truth(const pipeline::SceneSnapshot& s,
                                   const std::vector<eval::Pose>& truth);

}  // namespace sfm::viewer
