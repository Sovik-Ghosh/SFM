#include <cmath>

#include "sfm/ba/ba.hpp"

namespace sfm::ba {

ObsPose observation_pose(const Problem& p, const Observation& o) {
  const auto c = static_cast<std::size_t>(o.camera);
  if (o.sensor < 0) return {p.K[c], p.R[c], p.t[c]};
  const Sensor& s = p.sensors[static_cast<std::size_t>(o.sensor)];
  return {s.K, s.R * p.R[c], s.R * p.t[c] + s.t};
}

Eigen::Vector2d residual(const Problem& p, const Observation& o) {
  const ObsPose q = observation_pose(p, o);
  const Eigen::Vector3d h = q.K * (q.R * p.X[static_cast<std::size_t>(o.point)] + q.t);
  return h.head<2>() / h.z() - o.uv;
}

void observation_jacobians(const Problem& p, const Observation& o, Eigen::Matrix<double, 2, 6>& Jc,
                           Eigen::Matrix<double, 2, 3>& Jx) {
  const ObsPose q = observation_pose(p, o);
  jacobians(q.K, q.R, q.t, p.X[static_cast<std::size_t>(o.point)], Jc, Jx);
  if (o.sensor >= 0) {
    const Eigen::Matrix3d& Rs = p.sensors[static_cast<std::size_t>(o.sensor)].R;
    Jc.leftCols<3>() = (Jc.leftCols<3>() * Rs).eval();
    Jc.rightCols<3>() = (Jc.rightCols<3>() * Rs).eval();
  }
}

double cost(const Problem& p, double huber_px) {
  double c = 0.0;
  for (const Observation& o : p.obs) {
    const double s = residual(p, o).norm();
    c += s <= huber_px ? s * s : 2.0 * huber_px * s - huber_px * huber_px;
  }
  return 0.5 * c;
}

NormalEquations build_normal_equations(const Problem& p, double huber_px) {
  NormalEquations ne;
  ne.U.assign(p.cameras(), Eigen::Matrix<double, 6, 6>::Zero());
  ne.V.assign(p.points(), Eigen::Matrix3d::Zero());
  ne.W.resize(p.obs.size());
  ne.gc.assign(p.cameras(), Eigen::Matrix<double, 6, 1>::Zero());
  ne.gp.assign(p.points(), Eigen::Vector3d::Zero());
  for (std::size_t k = 0; k < p.obs.size(); ++k) {
    const Observation& o = p.obs[k];
    const auto c = static_cast<std::size_t>(o.camera);
    const auto x = static_cast<std::size_t>(o.point);
    Eigen::Matrix<double, 2, 6> Jc;
    Eigen::Matrix<double, 2, 3> Jx;
    observation_jacobians(p, o, Jc, Jx);
    const Eigen::Vector2d r = residual(p, o);
    const double s = r.norm();
    const double w = s <= huber_px ? 1.0 : huber_px / s;  // IRLS weight of the Huber loss
    ne.U[c] += w * Jc.transpose() * Jc;
    ne.V[x] += w * Jx.transpose() * Jx;
    ne.W[k] = w * Jc.transpose() * Jx;
    ne.gc[c] += w * Jc.transpose() * r;
    ne.gp[x] += w * Jx.transpose() * r;
  }
  return ne;
}

}  // namespace sfm::ba
