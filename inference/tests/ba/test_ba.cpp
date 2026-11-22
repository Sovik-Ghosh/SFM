#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>

#include "sfm/ba/ba.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/geometry.hpp"

#include "rig_problem.hpp"

using sfm::ba::Observation;
using sfm::ba::Problem;

namespace {

// Problem from a synthetic scene; camera 0 fixed, t_x of camera 1 fixed (gauge).
Problem make_problem(const sfm::eval::SyntheticScene& s) {
  Problem p;
  for (const auto& pose : s.poses) {
    p.K.push_back(s.K);
    p.R.push_back(pose.R);
    p.t.push_back(pose.t);
    p.fixed.push_back({false, false, false, false, false, false});
  }
  p.fixed[0] = {true, true, true, true, true, true};
  p.fixed[1][3] = true;
  for (Eigen::Index k = 0; k < s.points.cols(); ++k) p.X.emplace_back(s.points.col(k));
  for (const auto& o : s.observations) p.obs.push_back({o.camera, o.point, o.uv});
  return p;
}

sfm::eval::SyntheticScene scene(double noise, int cameras = 5, int points = 60) {
  sfm::eval::SyntheticConfig c;
  c.cameras = cameras;
  c.points = points;
  c.box_half = 15.0;
  c.noise_px = noise;
  return sfm::eval::make_synthetic_scene(c);
}

using sfm::ba::residual;

void perturb(Problem& p, std::uint64_t seed, double rot, double trans, double pts) {
  sfm::eval::Rng rng(seed);
  for (std::size_t c = 0; c < p.cameras(); ++c) {
    Eigen::Vector3d w(rng.normal(), rng.normal(), rng.normal());
    Eigen::Vector3d dt(rng.normal(), rng.normal(), rng.normal());
    for (std::size_t k = 0; k < 3; ++k) {
      if (p.fixed[c][k]) w(static_cast<Eigen::Index>(k)) = 0;
      if (p.fixed[c][3 + k]) dt(static_cast<Eigen::Index>(k)) = 0;
    }
    p.R[c] = sfm::recon::exp_so3(rot * w) * p.R[c];
    p.t[c] += trans * dt;
  }
  for (auto& X : p.X) X += pts * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal());
}

}  // namespace

TEST(BaJacobian, MatchesCentralDifferences) {
  const auto s = scene(0.0);
  const Problem p = make_problem(s);
  const double h = 1e-6;
  for (std::size_t k = 0; k < p.obs.size(); k += 37) {
    const Observation& o = p.obs[k];
    const auto c = static_cast<std::size_t>(o.camera);
    const auto x = static_cast<std::size_t>(o.point);
    Eigen::Matrix<double, 2, 6> Jc;
    Eigen::Matrix<double, 2, 3> Jx;
    sfm::ba::jacobians(p.K[c], p.R[c], p.t[c], p.X[x], Jc, Jx);
    for (int d = 0; d < 9; ++d) {
      Problem a = p;
      Problem b = p;
      if (d < 3) {
        Eigen::Vector3d w = Eigen::Vector3d::Zero();
        w(d) = h;
        a.R[c] = sfm::recon::exp_so3(w) * p.R[c];
        b.R[c] = sfm::recon::exp_so3(-w) * p.R[c];
      } else if (d < 6) {
        a.t[c](d - 3) += h;
        b.t[c](d - 3) -= h;
      } else {
        a.X[x](d - 6) += h;
        b.X[x](d - 6) -= h;
      }
      const Eigen::Vector2d num = (residual(a, o) - residual(b, o)) / (2 * h);
      const Eigen::Vector2d ana =
          d < 6 ? Eigen::Vector2d(Jc.col(d)) : Eigen::Vector2d(Jx.col(d - 6));
      EXPECT_LT((num - ana).norm(), 1e-6 * std::max(1.0, ana.norm()))
          << "obs " << k << " param " << d;
    }
  }
}

TEST(BaSchur, EqualsTheDenseSolve) {
  auto s = scene(0.5, 4, 25);
  Problem p = make_problem(s);
  perturb(p, 5, 0.01, 0.2, 0.2);
  const double huber = 2.0;
  const double lambda = 0.1;
  const auto ne = sfm::ba::build_normal_equations(p, huber);
  std::vector<Eigen::Matrix<double, 6, 1>> dc;
  std::vector<Eigen::Vector3d> dp;
  ASSERT_TRUE(sfm::ba::schur_solve(p, ne, lambda, dc, dp));

  // Dense system over the free parameters (cameras, then points), same damping.
  std::vector<int> col;
  int n = 0;
  const std::size_t nc = p.cameras();
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t k = 0; k < 6; ++k) col.push_back(p.fixed[c][k] ? -1 : n++);
  }
  const int cams = n;
  n += static_cast<int>(3 * p.points());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
  for (std::size_t c = 0; c < nc; ++c) {
    for (int r = 0; r < 6; ++r) {
      const int i = col[6 * c + static_cast<std::size_t>(r)];
      if (i < 0) continue;
      g(i) = ne.gc[c](r);
      for (int q = 0; q < 6; ++q) {
        const int j = col[6 * c + static_cast<std::size_t>(q)];
        if (j >= 0) H(i, j) = ne.U[c](r, q) + (r == q ? lambda * ne.U[c](r, r) + 1e-12 : 0.0);
      }
    }
  }
  for (std::size_t x = 0; x < p.points(); ++x) {
    const int b = cams + 3 * static_cast<int>(x);
    H.block<3, 3>(b, b) = ne.V[x];
    for (int r = 0; r < 3; ++r) H(b + r, b + r) += lambda * ne.V[x](r, r) + 1e-12;
    g.segment<3>(b) = ne.gp[x];
  }
  for (std::size_t k = 0; k < p.obs.size(); ++k) {
    const auto c = static_cast<std::size_t>(p.obs[k].camera);
    const int b = cams + 3 * p.obs[k].point;
    for (int r = 0; r < 6; ++r) {
      const int i = col[6 * c + static_cast<std::size_t>(r)];
      if (i < 0) continue;
      H.block<1, 3>(i, b) += ne.W[k].row(r);
      H.block<3, 1>(b, i) += ne.W[k].row(r).transpose();
    }
  }
  const Eigen::VectorXd d = H.ldlt().solve(-g);
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t r = 0; r < 6; ++r) {
      const int i = col[6 * c + r];
      const double want = i < 0 ? 0.0 : d(i);
      EXPECT_NEAR(dc[c](static_cast<Eigen::Index>(r)), want, 1e-9 * std::max(1.0, std::fabs(want)));
    }
  }
  for (std::size_t x = 0; x < p.points(); ++x) {
    EXPECT_LT((dp[x] - d.segment<3>(cams + 3 * static_cast<int>(x))).norm(), 1e-9);
  }
}

TEST(BaLm, ConvergesToTheTruthAndNeverIncreasesTheCost) {
  const auto s = scene(0.0);
  Problem p = make_problem(s);
  perturb(p, 7, 0.02, 0.5, 0.3);
  const auto fixed_R = p.R[0];
  const auto fixed_t = p.t[0];
  const double fixed_tx = p.t[1].x();
  sfm::ba::Options o;
  o.max_iters = 100;
  const auto rep = sfm::ba::solve(p, o);
  EXPECT_TRUE(rep.converged);
  for (std::size_t k = 1; k < rep.cost_log.size(); ++k) {
    EXPECT_LT(rep.cost_log[k], rep.cost_log[k - 1]);
  }
  EXPECT_LT(rep.final_cost, 1e-12);
  EXPECT_EQ(p.R[0], fixed_R);  // gauge held
  EXPECT_EQ(p.t[0], fixed_t);
  EXPECT_EQ(p.t[1].x(), fixed_tx);
  for (std::size_t c = 0; c < p.cameras(); ++c) {
    EXPECT_LT(Eigen::AngleAxisd(p.R[c] * s.poses[c].R.transpose()).angle() * 180 / M_PI, 1e-6);
    EXPECT_LT((p.t[c] - s.poses[c].t).norm(), 1e-6);
  }
  for (std::size_t x = 0; x < p.points(); ++x) {
    EXPECT_LT((p.X[x] - s.points.col(static_cast<Eigen::Index>(x))).norm(), 1e-6);
  }
}

TEST(BaLm, HuberLimitsAnOutlier) {
  const auto s = scene(0.3);
  Problem p = make_problem(s);
  p.obs[10].uv += Eigen::Vector2d(80.0, -60.0);  // one gross outlier
  sfm::ba::Options o;
  const double huber_cost = sfm::ba::cost(p, 2.0);
  const double quad_cost = sfm::ba::cost(p, 1e9);
  EXPECT_LT(huber_cost, quad_cost);  // linear, not quadratic, in the 100 px error
  sfm::ba::solve(p, o);
  // The outlier still reprojects far away: Huber did not let it drag the solution onto it.
  EXPECT_GT(residual(p, p.obs[10]).norm(), 90.0);
}

TEST(BaRig, JacobianMatchesCentralDifferencesOfTheFramePose) {
  const Problem p = sfm::test::make_rig_problem(scene(0.0));
  const double h = 1e-6;
  for (std::size_t k = 0; k < p.obs.size(); k += 23) {
    const Observation& o = p.obs[k];
    const auto c = static_cast<std::size_t>(o.camera);
    const auto x = static_cast<std::size_t>(o.point);
    Eigen::Matrix<double, 2, 6> Jc;
    Eigen::Matrix<double, 2, 3> Jx;
    sfm::ba::observation_jacobians(p, o, Jc, Jx);
    for (int d = 0; d < 9; ++d) {
      Problem a = p;
      Problem b = p;
      if (d < 3) {
        Eigen::Vector3d w = Eigen::Vector3d::Zero();
        w(d) = h;
        a.R[c] = sfm::recon::exp_so3(w) * p.R[c];
        b.R[c] = sfm::recon::exp_so3(-w) * p.R[c];
      } else if (d < 6) {
        a.t[c](d - 3) += h;
        b.t[c](d - 3) -= h;
      } else {
        a.X[x](d - 6) += h;
        b.X[x](d - 6) -= h;
      }
      const Eigen::Vector2d num = (sfm::ba::residual(a, o) - sfm::ba::residual(b, o)) / (2 * h);
      const Eigen::Vector2d ana =
          d < 6 ? Eigen::Vector2d(Jc.col(d)) : Eigen::Vector2d(Jx.col(d - 6));
      EXPECT_LT((num - ana).norm(), 1e-6 * std::max(1.0, ana.norm()))
          << "obs " << k << " sensor " << o.sensor << " param " << d;
    }
  }
}

TEST(BaRig, IdentitySensorEqualsNoSensor) {
  // A sensor with the identity offset and the camera's K must give the same normal equations,
  // bit for bit, as the plain observation (P9a problems are unchanged).
  const auto s = scene(0.5);
  Problem plain = make_problem(s);
  Problem rig = plain;
  rig.sensors.push_back({s.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()});
  for (auto& o : rig.obs) o.sensor = 0;
  perturb(plain, 3, 0.01, 0.2, 0.1);
  rig.R = plain.R;
  rig.t = plain.t;
  rig.X = plain.X;
  const auto a = sfm::ba::build_normal_equations(plain, 2.0);
  const auto b = sfm::ba::build_normal_equations(rig, 2.0);
  for (std::size_t c = 0; c < plain.cameras(); ++c) {
    EXPECT_EQ(a.U[c], b.U[c]);
    EXPECT_EQ(a.gc[c], b.gc[c]);
  }
  for (std::size_t k = 0; k < plain.obs.size(); ++k) EXPECT_EQ(a.W[k], b.W[k]);
  EXPECT_EQ(sfm::ba::cost(plain, 2.0), sfm::ba::cost(rig, 2.0));
}

TEST(BaRig, ConvergesToTheTruthWithOnePosePerFrame) {
  const auto s = scene(0.0);
  Problem p = sfm::test::make_rig_problem(s);
  const Problem truth = p;
  perturb(p, 11, 0.02, 0.5, 0.3);
  sfm::ba::Options o;
  o.max_iters = 100;
  const auto rep = sfm::ba::solve(p, o);
  EXPECT_TRUE(rep.converged);
  EXPECT_LT(rep.final_cost, 1e-12);
  for (std::size_t c = 0; c < p.cameras(); ++c) {
    EXPECT_LT(Eigen::AngleAxisd(p.R[c] * truth.R[c].transpose()).angle() * 180 / M_PI, 1e-6);
    EXPECT_LT((p.t[c] - truth.t[c]).norm(), 1e-6);
  }
}
