#pragma once
// Synthetic scenes as pipeline inputs: perfect features, pairs and tracks (keypoint k of every
// image = point k).
#include <cstddef>
#include <vector>

#include "sfm/eval/synthetic.hpp"
#include "sfm/features/features.hpp"
#include "sfm/matching/fundamental.hpp"
#include "sfm/matching/matches.hpp"
#include "sfm/matching/verify.hpp"

namespace sfm::recon::test {

struct PipelineInput {
  eval::SyntheticScene scene;
  std::vector<features::Features> features;
  std::vector<matching::PairMatches> pairs;
  matching::Tracks tracks;
  std::vector<Eigen::Matrix3d> K;
};

// Inputs from a scene whose observations are camera-major (index = camera * points + point).
inline PipelineInput make_input(const eval::SyntheticScene& scene) {
  PipelineInput in;
  in.scene = scene;
  const std::size_t nc = scene.poses.size();
  const auto np = static_cast<std::size_t>(scene.points.cols());
  eval::SyntheticConfig cfg;
  cfg.cameras = static_cast<int>(nc);
  in.features.resize(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t p = 0; p < np; ++p) {
      const Eigen::Vector2d uv = in.scene.observations[c * np + p].uv;
      in.features[c].xy.push_back(static_cast<float>(uv.x()));
      in.features[c].xy.push_back(static_cast<float>(uv.y()));
      in.features[c].angle.push_back(0.0F);
    }
  }
  for (int i = 0; i < cfg.cameras; ++i) {
    for (int j = i + 1; j < cfg.cameras; ++j) {
      matching::PairMatches pm;
      pm.i = i;
      pm.j = j;
      matching::Points2 a;
      matching::Points2 b;
      for (std::size_t p = 0; p < np; ++p) {
        pm.idx.push_back(static_cast<std::int32_t>(p));
        pm.idx.push_back(static_cast<std::int32_t>(p));
        const auto& fi = in.features[static_cast<std::size_t>(i)];
        const auto& fj = in.features[static_cast<std::size_t>(j)];
        a.emplace_back(fi.xy[2 * p], fi.xy[2 * p + 1]);
        b.emplace_back(fj.xy[2 * p], fj.xy[2 * p + 1]);
      }
      pm.F = matching::eight_point(a, b);
      in.pairs.push_back(pm);
    }
  }
  in.tracks = matching::build_tracks(in.pairs, std::vector<std::size_t>(nc, np));
  in.K.assign(nc, in.scene.K);
  return in;
}

inline PipelineInput make_input(const eval::SyntheticConfig& cfg) {
  return make_input(eval::make_synthetic_scene(cfg));
}

}  // namespace sfm::recon::test
