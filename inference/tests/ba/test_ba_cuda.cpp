// GPU bundle adjustment against the CPU implementation.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

#include "sfm/ba/ba.hpp"
#include "sfm/ba/ba_cuda.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/geometry.hpp"

#include "rig_problem.hpp"
#include "test_env.hpp"

using namespace sfm::ba;  // NOLINT(google-build-using-namespace)

namespace {

Problem make_problem(double noise, int cameras, int points, std::uint64_t seed) {
  sfm::eval::SyntheticConfig c;
  c.cameras = cameras;
  c.points = points;
  c.box_half = 15.0;
  c.noise_px = noise;
  c.seed = seed;
  const auto s = sfm::eval::make_synthetic_scene(c);
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
  // Drop every 7th observation so cameras do not all see all points (sparser structure).
  for (std::size_t k = 0; k < s.observations.size(); ++k) {
    if (k % 7 == 3) continue;
    const auto& o = s.observations[k];
    p.obs.push_back({o.camera, o.point, o.uv});
  }
  return p;
}

void perturb(Problem& p, std::uint64_t seed) {
  sfm::eval::Rng rng(seed);
  for (std::size_t c = 1; c < p.cameras(); ++c) {
    p.R[c] = sfm::recon::exp_so3(0.01 * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal())) *
             p.R[c];
    for (int k = 0; k < 3; ++k) {
      if (!p.fixed[c][3 + static_cast<std::size_t>(k)]) p.t[c](k) += 0.3 * rng.normal();
    }
  }
  for (auto& X : p.X) X += 0.2 * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal());
  p.obs[5].uv += Eigen::Vector2d(40, -30);  // one Huber outlier
}

template <class A, class B>
double rel(const A& a, const B& b) {
  return (a - b).norm() / std::max(1.0, b.norm());
}

}  // namespace

TEST(BaCuda, NormalEquationsAndCostEqualTheCpu) {
  Problem p = make_problem(0.5, 6, 80, 3);
  perturb(p, 4);
  CudaBackend gpu(p);
  EXPECT_NEAR(gpu.cost(p, 2.0), cost(p, 2.0), 1e-9 * cost(p, 2.0));
  gpu.linearize(p, 2.0);
  const NormalEquations g = gpu.normal_equations();
  const NormalEquations c = build_normal_equations(p, 2.0);
  for (std::size_t k = 0; k < p.cameras(); ++k) {
    EXPECT_LT(rel(g.U[k], c.U[k]), 1e-12) << k;
    EXPECT_LT(rel(g.gc[k], c.gc[k]), 1e-12) << k;
  }
  for (std::size_t k = 0; k < p.points(); ++k) {
    EXPECT_LT(rel(g.V[k], c.V[k]), 1e-12) << k;
    EXPECT_LT(rel(g.gp[k], c.gp[k]), 1e-12) << k;
  }
  for (std::size_t k = 0; k < p.obs.size(); ++k) EXPECT_LT(rel(g.W[k], c.W[k]), 1e-12) << k;
}

TEST(BaCuda, SchurSolveEqualsTheCpu) {
  Problem p = make_problem(0.5, 7, 120, 5);
  perturb(p, 6);
  CudaBackend gpu(p);
  gpu.linearize(p, 2.0);
  const NormalEquations ne = build_normal_equations(p, 2.0);
  for (const double lambda : {1e-6, 1e-2, 10.0}) {
    std::vector<Eigen::Matrix<double, 6, 1>> dc_g;
    std::vector<Eigen::Vector3d> dp_g;
    std::vector<Eigen::Matrix<double, 6, 1>> dc_c;
    std::vector<Eigen::Vector3d> dp_c;
    ASSERT_TRUE(gpu.solve(p, lambda, dc_g, dp_g));
    ASSERT_TRUE(schur_solve(p, ne, lambda, dc_c, dp_c));
    for (std::size_t k = 0; k < p.cameras(); ++k) EXPECT_LT(rel(dc_g[k], dc_c[k]), 1e-8) << lambda;
    for (std::size_t k = 0; k < p.points(); ++k) EXPECT_LT(rel(dp_g[k], dp_c[k]), 1e-8) << lambda;
    EXPECT_EQ(dc_g[0].norm(), 0.0);  // fixed camera
    EXPECT_EQ(dc_g[1](3), 0.0);      // fixed scale component
  }
}

TEST(BaCuda, ConvergesLikeTheCpuAndIsDeterministic) {
  Problem p0 = make_problem(0.0, 6, 100, 7);
  const Problem truth = p0;
  perturb(p0, 8);
  p0.obs[5].uv -= Eigen::Vector2d(40, -30);  // noise-free: no outlier
  Options o;
  o.max_iters = 100;
  Problem a = p0;
  Problem b = p0;
  Problem c = p0;
  const Report ra = solve_cuda(a, o);
  const Report rb = solve_cuda(b, o);
  const Report rc = solve(c, o);
  EXPECT_TRUE(ra.converged);
  EXPECT_LT(ra.final_cost, 1e-12);
  for (std::size_t k = 1; k < ra.cost_log.size(); ++k) {
    EXPECT_LT(ra.cost_log[k], ra.cost_log[k - 1]);
  }
  EXPECT_EQ(ra.cost_log, rb.cost_log);  // deterministic
  for (std::size_t k = 0; k < a.cameras(); ++k) {
    EXPECT_EQ(a.R[k], b.R[k]);
    EXPECT_LT(Eigen::AngleAxisd(a.R[k] * truth.R[k].transpose()).angle() * 180 / M_PI, 1e-6);
    EXPECT_LT((a.t[k] - truth.t[k]).norm(), 1e-6);
    EXPECT_LT((a.t[k] - c.t[k]).norm(), 1e-6);  // same solution as the CPU
  }
  EXPECT_EQ(a.R[0], p0.R[0]);  // gauge held
  EXPECT_EQ(a.t[1].x(), p0.t[1].x());
  EXPECT_TRUE(
      rc.converged);  // iteration counts differ near cost 0, where the stop test sees rounding
}

// A large reduced system (300 cameras, 1793 unknowns: past the one-block kernel's 40 KB column)
// is solved by cuSOLVER (P9a: the 4-camera rig scenes have about 1000 cameras).
TEST(BaCuda, LargeReducedSystemEqualsTheCpu) {
  // 40 cameras still take the cuSOLVER path (> 16 free cameras) under the sanitizer.
  Problem p =
      make_problem(0.5, sfm::test::sanitizer_size(300, 40), sfm::test::sanitizer_size(200, 60), 11);
  perturb(p, 12);
  CudaBackend gpu(p);
  gpu.linearize(p, 2.0);
  const NormalEquations ne = build_normal_equations(p, 2.0);
  for (const double lambda : {1e-6, 1e-2}) {
    std::vector<Eigen::Matrix<double, 6, 1>> dc_g;
    std::vector<Eigen::Vector3d> dp_g;
    std::vector<Eigen::Matrix<double, 6, 1>> dc_c;
    std::vector<Eigen::Vector3d> dp_c;
    ASSERT_TRUE(gpu.solve(p, lambda, dc_g, dp_g));
    ASSERT_TRUE(schur_solve(p, ne, lambda, dc_c, dp_c));
    for (std::size_t k = 0; k < p.cameras(); ++k) EXPECT_LT(rel(dc_g[k], dc_c[k]), 1e-8) << k;
    for (std::size_t k = 0; k < p.points(); ++k) EXPECT_LT(rel(dp_g[k], dp_c[k]), 1e-8) << k;
    EXPECT_EQ(dc_g[0].norm(), 0.0);  // fixed camera
  }
  // Determinism: the same solve twice gives the same bits.
  std::vector<Eigen::Matrix<double, 6, 1>> a;
  std::vector<Eigen::Matrix<double, 6, 1>> b;
  std::vector<Eigen::Vector3d> pa;
  std::vector<Eigen::Vector3d> pb;
  ASSERT_TRUE(gpu.solve(p, 1e-3, a, pa));
  ASSERT_TRUE(gpu.solve(p, 1e-3, b, pb));
  EXPECT_EQ(a, b);
  EXPECT_EQ(pa, pb);
}

// The Schur term list grows with the square of the views per point: a problem whose list would
// not fit is refused with a clear error before anything is allocated (P9a: the synthetic
// benchmark at 576 cameras grew to > 30 GB of host memory and was killed).
TEST(BaCuda, RefusesATermListThatDoesNotFit) {
  Problem p;
  // One point seen by all: 50000^2 = 2.5e9 terms, over the int offsets on any host. The cameras
  // are free: pairs with a fixed camera have no terms (P9c), so all-fixed cameras would give none.
  constexpr int kCameras = 50000;
  for (int c = 0; c < kCameras; ++c) {
    p.K.push_back(Eigen::Matrix3d::Identity());
    p.R.push_back(Eigen::Matrix3d::Identity());
    p.t.emplace_back(c, 0.0, 0.0);
    p.fixed.push_back({false, false, false, false, false, false});
    p.obs.push_back({c, 0, Eigen::Vector2d::Zero()});
  }
  p.X.emplace_back(0.0, 0.0, 10.0);
  try {
    CudaBackend gpu(p);
    FAIL() << "expected a refusal";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("Schur terms"), std::string::npos) << e.what();
  }
}

TEST(BaCudaRig, NormalEquationsSolveAndConvergenceEqualTheCpu) {
  // Frames of a 3-sensor rig (sfm/ba/ba.hpp, P9b): the GPU composes the sensor offsets like the
  // CPU.
  sfm::eval::SyntheticConfig c;
  c.cameras = 6;
  c.points = 70;
  c.box_half = 15.0;
  c.seed = 21;
  Problem p = sfm::test::make_rig_problem(sfm::eval::make_synthetic_scene(c));
  const Problem truth = p;
  perturb(p, 22);
  CudaBackend gpu(p);
  EXPECT_NEAR(gpu.cost(p, 2.0), cost(p, 2.0), 1e-9 * cost(p, 2.0));
  gpu.linearize(p, 2.0);
  const NormalEquations g = gpu.normal_equations();
  const NormalEquations ne = build_normal_equations(p, 2.0);
  for (std::size_t k = 0; k < p.cameras(); ++k) {
    EXPECT_LT(rel(g.U[k], ne.U[k]), 1e-12) << k;
    EXPECT_LT(rel(g.gc[k], ne.gc[k]), 1e-12) << k;
  }
  for (std::size_t k = 0; k < p.obs.size(); ++k) EXPECT_LT(rel(g.W[k], ne.W[k]), 1e-12) << k;
  std::vector<Eigen::Matrix<double, 6, 1>> dc_g;
  std::vector<Eigen::Vector3d> dp_g;
  std::vector<Eigen::Matrix<double, 6, 1>> dc_c;
  std::vector<Eigen::Vector3d> dp_c;
  ASSERT_TRUE(gpu.solve(p, 1e-3, dc_g, dp_g));
  ASSERT_TRUE(schur_solve(p, ne, 1e-3, dc_c, dp_c));
  for (std::size_t k = 0; k < p.cameras(); ++k) EXPECT_LT(rel(dc_g[k], dc_c[k]), 1e-8) << k;
  // Both converge to the same poses; the 40 px outlier perturb() adds leaves them a few
  // thousandths of a degree from the truth (Huber bounds its pull, it does not remove it).
  Options o;
  o.max_iters = 100;
  Problem q = p;
  solve_cuda(p, o);
  solve(q, o);
  for (std::size_t k = 0; k < p.cameras(); ++k) {
    EXPECT_LT(Eigen::AngleAxisd(p.R[k] * q.R[k].transpose()).angle() * 180 / M_PI, 1e-7) << k;
    EXPECT_LT(rel(p.t[k], q.t[k]), 1e-7) << k;
    EXPECT_LT(Eigen::AngleAxisd(p.R[k] * truth.R[k].transpose()).angle() * 180 / M_PI, 0.01) << k;
  }
}
