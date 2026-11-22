#include "sfm/viewer/ground_truth.hpp"

#include <Eigen/LU>

#include "sfm/core/error.hpp"
#include "sfm/eval/pose_error.hpp"

namespace sfm::viewer {

GroundTruthView align_ground_truth(const pipeline::SceneSnapshot& s,
                                   const std::vector<eval::Pose>& truth) {
  GroundTruthView g;
  const std::size_t n = s.registered.size();
  g.aligned.assign(n, eval::Pose{});
  g.rotation_deg.assign(n, -1.0);
  g.centre_pct.assign(n, -1.0);
  std::vector<eval::Pose> est;
  std::vector<eval::Pose> tru;
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < n && i < truth.size(); ++i) {
    if (s.registered[i] == 0) continue;
    est.push_back({s.R[i], s.t[i]});
    tru.push_back(truth[i]);
    idx.push_back(i);
  }
  if (est.size() < 3) return g;
  eval::PoseErrors e;
  try {
    e = eval::pose_error(est, tru);
  } catch (const GeometryError&) {
    return g;  // collinear centres: alignment undetermined
  }
  // pose_error maps the estimate into the true frame: X_true = s R X_est + t. The inverse maps a
  // true pose into the estimate's frame: X_est = R^T (X_true - t) / s, so a camera
  // x = R_t X_true + t_t becomes x = (R_t R) X_est + (R_t t + t_t) / s, up to the scale of x:
  // x / s = R_t R X_est + (R_t t + t_t) / s.
  const auto& a = e.alignment;
  for (std::size_t k = 0; k < idx.size(); ++k) {
    const eval::Pose& t = tru[k];
    g.aligned[idx[k]] = {t.R * a.R, (t.R * a.t + t.t) / a.s};
    g.rotation_deg[idx[k]] = e.rotation_deg[k];
    g.centre_pct[idx[k]] = e.centre_pct[k];
  }
  g.valid = true;
  return g;
}

}  // namespace sfm::viewer
