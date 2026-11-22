#pragma once
// Camera rig of a dataset (P9b): sensors fixed to each other, triggered together. Image i shows
// sensor s at moment ("frame") f; its pose is T_off[s] * T_frame[f] (x_s = R_s x_frame + t_s).
// Read from <dataset root>/rig.txt (written by tools/rig_from_dataset.py):
//   SENSOR <s> <name> r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
//   IMAGE <id> <frame> <sensor>
// Lines starting with '#' are comments.

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace sfm::data {

struct Rig {
  std::vector<int> frame;          // per loaded image (index into Dataset::ids): 0..frames()-1
  std::vector<int> sensor;         // per loaded image
  std::vector<Eigen::Matrix3d> R;  // per sensor
  std::vector<Eigen::Vector3d> t;  // per sensor
  int frames = 0;

  bool empty() const { return frame.empty(); }
  std::size_t sensors() const { return R.size(); }
};

// Rig of the images `ids` (in that order). Frames are renumbered 0.. in order of first appearance
// in `ids`, so a subset of a dataset (first..last) gives consecutive frames. Throws IoError on a
// malformed line, an id without an IMAGE line, an unknown sensor, or a sensor rotation that is
// not a rotation (|R^T R - I| > 1e-6).
Rig parse_rig(const std::string& text, const std::vector<int>& ids, const std::string& name);
Rig load_rig(const std::filesystem::path& file, const std::vector<int>& ids);

}  // namespace sfm::data
