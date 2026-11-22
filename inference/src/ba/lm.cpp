#include <algorithm>
#include <cmath>

#include "sfm/ba/ba.hpp"
#include "sfm/core/trace.hpp"

namespace sfm::ba {

namespace {

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
  const double theta = w.norm();
  Eigen::Matrix3d W;
  W << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
  if (theta < 1e-8) return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
  return Eigen::Matrix3d::Identity() + std::sin(theta) / theta * W +
         (1 - std::cos(theta)) / (theta * theta) * W * W;
}

class CpuBackend final : public LmBackend {
 public:
  double cost(const Problem& p, double huber_px) override { return ba::cost(p, huber_px); }
  void linearize(const Problem& p, double huber_px) override {
    ne_ = build_normal_equations(p, huber_px);
  }
  bool solve(const Problem& p, double lambda, std::vector<Eigen::Matrix<double, 6, 1>>& dc,
             std::vector<Eigen::Vector3d>& dp) override {
    return schur_solve(p, ne_, lambda, dc, dp);
  }
  // Predicted cost decrease of the step d for the quadratic model with gradient g and damped
  // system (H + lambda D) d = -g:  L(0) - L(d) = -g.d - 1/2 d.H.d = 1/2 (-g.d + lambda d.D.d).
  double predicted_decrease(double lambda, const std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                            const std::vector<Eigen::Vector3d>& dp) override {
    double gd = 0.0;
    double dDd = 0.0;
    for (std::size_t c = 0; c < dc.size(); ++c) {
      gd += ne_.gc[c].dot(dc[c]);
      dDd += dc[c].cwiseProduct(ne_.U[c].diagonal()).dot(dc[c]);
    }
    for (std::size_t x = 0; x < dp.size(); ++x) {
      gd += ne_.gp[x].dot(dp[x]);
      dDd += dp[x].cwiseProduct(ne_.V[x].diagonal()).dot(dp[x]);
    }
    return 0.5 * (-gd + lambda * dDd);
  }

 private:
  NormalEquations ne_;
};

}  // namespace

Report solve(Problem& p, const Options& o) {
  CpuBackend backend;
  return solve_lm(p, o, backend);
}

Report solve_lm(Problem& p, const Options& o, LmBackend& backend) {
  Report rep;
  double current = backend.cost(p, o.huber_px);
  rep.initial_cost = current;
  rep.cost_log.push_back(current);
  // Damping schedule of Nielsen (1999): after a step with gain ratio rho = actual / predicted
  // decrease, lambda *= max(1/3, 1 - (2 rho - 1)^3) if accepted, else lambda *= nu, nu *= 2.
  double lambda = o.initial_lambda;
  double nu = 2.0;
  std::vector<Eigen::Matrix<double, 6, 1>> dc;
  std::vector<Eigen::Vector3d> dp;
  {
    const core::TraceRange r("ba.linearize");
    backend.linearize(p, o.huber_px);
  }
  while (rep.iterations < o.max_iters) {
    ++rep.iterations;
    bool solved = false;
    {
      const core::TraceRange r("ba.solve");
      solved = backend.solve(p, lambda, dc, dp);
    }
    if (!solved) {
      lambda *= nu;
      nu *= 2.0;
      if (lambda > 1e16) break;
      continue;
    }
    Problem trial = p;
    for (std::size_t c = 0; c < p.cameras(); ++c) {
      trial.R[c] = exp_so3(dc[c].head<3>()) * p.R[c];
      trial.t[c] = p.t[c] + dc[c].tail<3>();
    }
    for (std::size_t x = 0; x < p.points(); ++x) trial.X[x] = p.X[x] + dp[x];
    double next = 0.0;
    {
      const core::TraceRange r("ba.cost");
      next = backend.cost(trial, o.huber_px);
    }
    const double predicted = backend.predicted_decrease(lambda, dc, dp);
    const double rho = predicted > 0.0 ? (current - next) / predicted : -1.0;
    if (std::isfinite(next) && next < current && rho > 0.0) {
      const double decrease = (current - next) / current;
      p.R.swap(trial.R);
      p.t.swap(trial.t);
      p.X.swap(trial.X);
      current = next;
      rep.cost_log.push_back(current);
      ++rep.accepted;
      const double r = 2.0 * rho - 1.0;
      lambda *= std::max(1.0 / 3.0, 1.0 - r * r * r);
      nu = 2.0;
      if (decrease < o.function_tolerance) {
        rep.converged = true;
        break;
      }
      const core::TraceRange lin("ba.linearize");
      backend.linearize(p, o.huber_px);
    } else {
      lambda *= nu;
      nu *= 2.0;
      if (lambda > 1e16) {
        rep.converged = true;  // no step lowers the cost: at a minimum to working precision
        break;
      }
    }
  }
  rep.final_cost = current;
  return rep;
}

}  // namespace sfm::ba
