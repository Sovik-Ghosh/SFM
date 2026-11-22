#include "sfm/viewer/run_loader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "sfm/core/error.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/krt.hpp"
#include "sfm/pipeline/stage_io.hpp"

namespace sfm::viewer {

namespace fs = std::filesystem;

namespace {

std::string id_name(int id) {
  const std::string s = std::to_string(id);
  return std::string(s.size() < 4 ? 4 - s.size() : 0, '0') + s;
}

std::string read_text(const fs::path& f) {
  std::ifstream in(f);
  if (!in) return "";
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

std::vector<eval::Pose> load_truth(const config::DatasetConfig& d) {
  std::vector<eval::Pose> out;
  for (int id = d.first; id <= d.last; ++id) {
    const auto krt = data::decompose_projection(
        data::read_projection(d.root / d.calib_dir / (id_name(id) + ".txt")));
    out.push_back({krt.R, krt.t});
  }
  return out;
}

CameraGroups camera_groups(const std::string& names_txt, int first, int last) {
  CameraGroups out;
  if (last < first) return out;
  std::vector<std::string> folder(static_cast<std::size_t>(last - first + 1));
  std::istringstream in(names_txt);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    int id = -1;
    std::string name;
    if (!(ls >> id >> name) || id < first || id > last) continue;
    const auto slash = name.rfind('/');
    folder[static_cast<std::size_t>(id - first)] =
        slash == std::string::npos ? "" : name.substr(0, slash);
  }
  std::vector<std::string> keys;
  for (const auto& f : folder) {
    if (f.empty()) return {};  // an image without a folder: no grouping
    keys.push_back(f);
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  if (keys.size() < 2) return {};
  for (const auto& f : folder) {
    out.group.push_back(
        static_cast<int>(std::lower_bound(keys.begin(), keys.end(), f) - keys.begin()));
  }
  for (const auto& k : keys) {  // "images_rig_cam4_undistorted" -> "cam4"
    const auto at = k.find("cam");
    std::size_t end = at == std::string::npos ? at : at + 3;
    while (end != std::string::npos && end < k.size() &&
           std::isdigit(static_cast<unsigned char>(k[end])) != 0) {
      ++end;
    }
    out.labels.push_back(at != std::string::npos && end > at + 3 ? k.substr(at, end - at) : k);
  }
  return out;
}

CameraGroups load_camera_groups(const config::DatasetConfig& d) {
  return camera_groups(read_text(d.root / "names.txt"), d.first, d.last);
}

fs::path image_path(const config::DatasetConfig& d, int id) {
  return d.root / d.images_dir / (id_name(id) + ".ppm");
}

std::vector<pipeline::SceneSnapshot> replay_snapshots(const recon::Reconstruction& rec,
                                                      const recon::TrackIndex& index,
                                                      const std::vector<int>& ids,
                                                      const std::vector<std::uint8_t>& gray) {
  std::vector<pipeline::SceneSnapshot> out;
  const int n = static_cast<int>(rec.order.size());
  if (n < 2) return out;
  const int steps = std::min(n - 1, kMaxReplaySteps);
  int seq = 0;
  int prev = 0;
  for (int s = 0; s < steps; ++s) {
    // Camera counts 2 .. n, spread evenly over the steps (the last is always n).
    const int k = 2 + static_cast<int>(static_cast<long long>(n - 2) * (s + 1) / steps);
    if (k == prev) continue;
    prev = k;
    recon::Reconstruction part = rec;
    part.order.assign(rec.order.begin(), rec.order.begin() + k);
    for (auto& c : part.cameras) c.registered = false;
    for (const int i : part.order) part.cameras[static_cast<std::size_t>(i)].registered = true;
    for (int t = 0; t < index.track_count(); ++t) {
      if (part.has_point[static_cast<std::size_t>(t)] == 0) continue;
      int seen = 0;
      for (const recon::TrackObs* o = index.begin(t); o != index.end(t); ++o) {
        seen += part.cameras[static_cast<std::size_t>(o->image)].registered ? 1 : 0;
      }
      if (seen < 2) part.has_point[static_cast<std::size_t>(t)] = 0;
    }
    out.push_back(pipeline::make_snapshot(part, index, ids, gray, seq++));
  }
  return out;
}

LoadedRun load_run(const fs::path& run_dir) {
  const std::string cfg_text = read_text(run_dir / "config.json");
  if (cfg_text.empty()) {
    throw IoError(run_dir.string() + ": no config.json (not an sfm_run output)");
  }
  LoadedRun r;
  r.config = config::parse_config(cfg_text);
  const fs::path st = run_dir / "stages";
  for (const char* sub : {"features", "tracks", "reconstruction"}) {
    if (!fs::is_directory(st / sub)) {
      throw IoError(run_dir.string() + ": no stages/" + sub +
                    " (run sfm_run with \"output\": {\"dump_stages\": true})");
    }
  }
  const config::DatasetConfig& d = r.config.dataset;
  std::vector<int> ids;
  std::vector<Eigen::Matrix3d> K;
  std::vector<data::Image8> gray;
  for (int id = d.first; id <= d.last; ++id) {
    ids.push_back(id);
    const auto krt = data::decompose_projection(
        data::read_projection(d.root / d.calib_dir / (id_name(id) + ".txt")));
    K.push_back(krt.K);
    r.truth.push_back({krt.R, krt.t});
    gray.push_back(data::load_gray(image_path(d, id)));
  }
  const auto features = pipeline::load_features(st / "features", ids.size());
  const auto tracks = pipeline::load_tracks(st / "tracks");
  const recon::TrackIndex index(tracks, features);
  const auto rec = pipeline::load_reconstruction(st / "reconstruction", K);
  const std::vector<std::uint8_t> tg = pipeline::track_gray(index, gray);
  r.snapshot = pipeline::make_snapshot(rec, index, ids, tg, 0);
  r.replay = replay_snapshots(rec, index, ids, tg);
  r.groups = load_camera_groups(d);
  r.metrics_json = read_text(run_dir / "metrics.json");
  r.timings_json = read_text(run_dir / "timings.json");
  return r;
}

}  // namespace sfm::viewer
