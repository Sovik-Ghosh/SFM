#include "sfm/recon/reconstruction.hpp"

#include <algorithm>
#include <stdexcept>

namespace sfm::recon {

int Reconstruction::registered_count() const {
  return static_cast<int>(
      std::count_if(cameras.begin(), cameras.end(), [](const Camera& c) { return c.registered; }));
}

int Reconstruction::point_count() const {
  return static_cast<int>(std::count(has_point.begin(), has_point.end(), std::uint8_t{1}));
}

TrackIndex::TrackIndex(const matching::Tracks& tracks,
                       const std::vector<features::Features>& features)
    : offset_(static_cast<std::size_t>(tracks.count) + 1, 0), by_image_(features.size()) {
  obs_.reserve(tracks.size());
  for (std::size_t k = 0; k < tracks.size(); ++k) {
    const int t = tracks.obs[3 * k];
    const int image = tracks.obs[3 * k + 1];
    const int kp = tracks.obs[3 * k + 2];
    if (t < 0 || t >= tracks.count || image < 0 ||
        static_cast<std::size_t>(image) >= features.size() || kp < 0 ||
        static_cast<std::size_t>(kp) >= features[static_cast<std::size_t>(image)].size()) {
      throw std::out_of_range("TrackIndex: observation index out of range");
    }
    if (k > 0 && t < tracks.obs[3 * (k - 1)]) {
      throw std::invalid_argument("TrackIndex: obs not sorted");
    }
    const auto& f = features[static_cast<std::size_t>(image)];
    const auto kk = static_cast<std::size_t>(kp);
    obs_.push_back({image, kp, Eigen::Vector2d(f.xy[2 * kk], f.xy[2 * kk + 1])});
    ++offset_[static_cast<std::size_t>(t) + 1];
    by_image_[static_cast<std::size_t>(image)].emplace_back(t, kp);
  }
  for (std::size_t t = 1; t < offset_.size(); ++t) offset_[t] += offset_[t - 1];
  for (auto& v : by_image_) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  }
}

bool observation_used(const Reconstruction& rec, const TrackIndex& index, const TrackObs* o) {
  if (!rec.cameras[static_cast<std::size_t>(o->image)].registered) return false;
  const std::size_t k = index.index_of(o);
  return k >= rec.obs_outlier.size() || rec.obs_outlier[k] == 0;
}

std::vector<PointObs> point_observations(const Reconstruction& rec, const TrackIndex& index) {
  std::vector<PointObs> out;
  for (int t = 0; t < index.track_count(); ++t) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
      if (observation_used(rec, index, o)) {
        out.push_back({o->image, t, o->uv});
      }
    }
  }
  return out;
}

std::vector<double> reprojection_errors(const Reconstruction& rec, const TrackIndex& index) {
  std::vector<double> e;
  for (const PointObs& o : point_observations(rec, index)) {
    const Camera& c = rec.cameras[static_cast<std::size_t>(o.camera)];
    e.push_back(
        reprojection_error(c.K, c.pose, rec.points[static_cast<std::size_t>(o.track)], o.uv));
  }
  return e;
}

Pose rig_frame_pose(const data::Rig& rig, int image, const Pose& image_pose) {
  const auto s = static_cast<std::size_t>(rig.sensor[static_cast<std::size_t>(image)]);
  const Eigen::Matrix3d Rt = rig.R[s].transpose();
  return {Rt * image_pose.R, Rt * (image_pose.t - rig.t[s])};
}

Pose rig_image_pose(const data::Rig& rig, int image, const Pose& frame_pose) {
  const auto s = static_cast<std::size_t>(rig.sensor[static_cast<std::size_t>(image)]);
  return {rig.R[s] * frame_pose.R, rig.R[s] * frame_pose.t + rig.t[s]};
}

}  // namespace sfm::recon
