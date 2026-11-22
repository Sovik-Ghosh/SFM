#pragma once
// Loading a finished run (outputs/<run>/ written by sfm_run with output.dump_stages) for viewing,
// and the dataset's calibrated poses.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/eval/pose.hpp"
#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer {

// Calibrated (true) pose per image of the dataset range, from <root>/<calib_dir>/NNNN.txt.
std::vector<eval::Pose> load_truth(const config::DatasetConfig& d);

// Physical camera of each image of a multi-camera dataset (the ETH3D rig: cam4..cam7), from the
// dataset's names.txt ("NNNN <folder>/<file>", written by tools/import_eth3d.py). Images of one
// folder share a group (0, 1, .. in folder name order); `labels` gets a short name per group
// ("cam4" when the folder names a camera, else the folder). Empty when there is one folder only,
// or the file is missing or does not cover first..last.
struct CameraGroups {
  std::vector<int> group;           // per image first..last
  std::vector<std::string> labels;  // per group
};
CameraGroups camera_groups(const std::string& names_txt, int first, int last);
CameraGroups load_camera_groups(const config::DatasetConfig& d);

// Image file of dataset id `id` (<root>/<images_dir>/NNNN.ppm).
std::filesystem::path image_path(const config::DatasetConfig& d, int id);

struct LoadedRun {
  config::Config config;  // the run's effective config (config.json)
  pipeline::SceneSnapshot snapshot;
  // A replay rebuilt from the registration order (stages/reconstruction/order.npy): the final
  // poses with the first k registered cameras, k = 2 .. all (at most kMaxReplaySteps steps), and
  // the points seen by at least two of them; the last one equals `snapshot`. A finished run keeps
  // no intermediate states, so the poses are the final ones throughout.
  std::vector<pipeline::SceneSnapshot> replay;
  std::vector<eval::Pose> truth;  // per image
  CameraGroups groups;            // per image; empty for a one-camera dataset
  std::string metrics_json;       // empty if the run has none
  std::string timings_json;
};

inline constexpr int kMaxReplaySteps = 120;

// The replay of `rec` (see LoadedRun::replay).
std::vector<pipeline::SceneSnapshot> replay_snapshots(const recon::Reconstruction& rec,
                                                      const recon::TrackIndex& index,
                                                      const std::vector<int>& ids,
                                                      const std::vector<std::uint8_t>& gray);

// Throws IoError if run_dir lacks config.json or the features / tracks / reconstruction stage
// dumps (run sfm_run with "output": {"dump_stages": true}).
LoadedRun load_run(const std::filesystem::path& run_dir);

}  // namespace sfm::viewer
