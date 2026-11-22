#pragma once
// Synthetic rig bundle-adjustment problem shared by the CPU and GPU BA tests.

#include <array>

#include <Eigen/Core>

#include "sfm/ba/ba.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/geometry.hpp"

namespace sfm::test {

// A rig problem from a synthetic scene: its poses are the frames; 3 sensors with small rotations
// and a few-centimetre baseline (camera distance ~10x the box), every point seen by every sensor of
// every frame. Sensor 0 is the identity, as in a real rig (the reference camera).
inline sfm::ba::Problem make_rig_problem(const sfm::eval::SyntheticScene& s) {
  sfm::ba::Problem p;
  for (const auto& pose : s.poses) {
    p.K.push_back(s.K);
    p.R.push_back(pose.R);
    p.t.push_back(pose.t);
    p.fixed.push_back({false, false, false, false, false, false});
  }
  p.fixed[0] = {true, true, true, true, true, true};
  p.fixed[1][3] = true;
  for (Eigen::Index k = 0; k < s.points.cols(); ++k) p.X.emplace_back(s.points.col(k));
  const std::array<Eigen::Vector3d, 3> w = {Eigen::Vector3d::Zero(),
                                            Eigen::Vector3d(0.01, -0.03, 0.02),
                                            Eigen::Vector3d(-0.02, 0.01, 0.04)};
  const std::array<Eigen::Vector3d, 3> t = {
      Eigen::Vector3d::Zero(), Eigen::Vector3d(-0.6, 0.0, 0.05), Eigen::Vector3d(0.1, -0.2, 0.0)};
  for (std::size_t k = 0; k < 3; ++k) {
    sfm::ba::Sensor r;
    r.K = s.K;
    r.K(0, 0) *= 1.0 + 0.2 * static_cast<double>(k);  // different lenses
    r.K(1, 1) *= 1.0 + 0.2 * static_cast<double>(k);
    r.R = sfm::recon::exp_so3(w[k]);
    r.t = t[k];
    p.sensors.push_back(r);
  }
  for (std::size_t c = 0; c < p.cameras(); ++c) {
    for (std::size_t sn = 0; sn < 3; ++sn) {
      for (std::size_t x = 0; x < p.points(); ++x) {
        sfm::ba::Observation o{static_cast<int>(c), static_cast<int>(x), Eigen::Vector2d::Zero(),
                               static_cast<int>(sn)};
        o.uv = sfm::ba::residual(p, o);  // uv = 0 above: the residual is the projection
        p.obs.push_back(o);
      }
    }
  }
  return p;
}

}  // namespace sfm::test
