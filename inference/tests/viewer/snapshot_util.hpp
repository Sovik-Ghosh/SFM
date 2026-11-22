#pragma once
// A SceneSnapshot of a synthetic scene (exact poses), for the viewer-core tests.
#include <cstdint>

#include "sfm/eval/synthetic.hpp"
#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer::test {

inline pipeline::SceneSnapshot synthetic_snapshot(int cameras = 6, int points = 50,
                                                  std::uint64_t seed = 1) {
  eval::SyntheticConfig c;
  c.cameras = cameras;
  c.points = points;
  c.box_half = 10.0;
  c.seed = seed;
  const auto s = eval::make_synthetic_scene(c);
  pipeline::SceneSnapshot snap;
  for (int i = 0; i < cameras; ++i) {
    snap.ids.push_back(i);
    snap.K.push_back(s.K);
    snap.R.push_back(s.poses[static_cast<std::size_t>(i)].R);
    snap.t.push_back(s.poses[static_cast<std::size_t>(i)].t);
    snap.registered.push_back(1);
    snap.order.push_back(i);
  }
  for (int p = 0; p < points; ++p) {
    snap.points.push_back(s.points.col(p));
    snap.point_gray.push_back(static_cast<std::uint8_t>(p * 5));
    snap.point_error_px.push_back(0.1F * static_cast<float>(p));
    snap.point_track_length.push_back(2 + p % 4);
  }
  return snap;
}

}  // namespace sfm::viewer::test
