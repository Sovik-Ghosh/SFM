// CPU vs GPU bundle adjustment on synthetic problems of growing size: time per LM iteration
// (20 iterations, no early stop), same perturbed start for both.
//   bench_ba [--large] [--gpu-only]   (--large adds 576 and 1152 cameras, the size of the 4-camera
//   rig scenes; --gpu-only skips the CPU reference, single-threaded and slow at that size)
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/ba/ba.hpp"
#include "sfm/ba/ba_cuda.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/recon/geometry.hpp"

namespace {

sfm::ba::Problem make(int cameras, int points) {
  sfm::eval::SyntheticConfig c;
  c.cameras = cameras;
  c.points = points;
  c.box_half = 15.0;
  c.noise_px = 0.5;
  c.seed = 1;
  const auto s = sfm::eval::make_synthetic_scene(c);
  sfm::ba::Problem p;
  sfm::eval::Rng rng(2);
  for (std::size_t k = 0; k < s.poses.size(); ++k) {
    p.K.push_back(s.K);
    p.R.push_back(k == 0 ? s.poses[k].R
                         : sfm::recon::exp_so3(
                               0.005 * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal())) *
                               s.poses[k].R);
    p.t.push_back(s.poses[k].t);
    p.fixed.push_back({false, false, false, false, false, false});
  }
  p.fixed[0] = {true, true, true, true, true, true};
  p.fixed[1][3] = true;
  for (Eigen::Index k = 0; k < s.points.cols(); ++k) {
    p.X.emplace_back(s.points.col(k) +
                     0.1 * Eigen::Vector3d(rng.normal(), rng.normal(), rng.normal()));
  }
  // Up to 288 cameras: each point seen by ~1/3 of the cameras (the README's numbers). Larger:
  // each point seen by a band of kTrack consecutive cameras, as in a video (a point seen by a third
  // of 576 cameras needs ~18 GB of Schur terms on the host, more than any real track).
  constexpr int kTrack = 12;
  for (const auto& o : s.observations) {
    const bool keep =
        cameras <= 288
            ? (o.camera + o.point) % 3 == 0
            : (o.camera - static_cast<int>(static_cast<long long>(o.point) * cameras / points) +
               cameras) %
                      cameras <
                  kTrack;
    if (keep) p.obs.push_back({o.camera, o.point, o.uv});
  }
  return p;
}

double ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    bool large = false;
    bool gpu_only = false;
    for (int k = 1; k < argc; ++k) {
      large = large || std::string(argv[k]) == "--large";
      gpu_only = gpu_only || std::string(argv[k]) == "--gpu-only";
    }
    sfm::ba::Options o;
    o.max_iters = 20;
    o.function_tolerance = 0.0;
    std::printf("%8s %8s %8s | %12s %12s %8s\n", "cameras", "points", "obs", "cpu ms/iter",
                "cuda ms/iter", "speed-up");
    {  // warm-up: CUDA context
      auto p = make(6, 100);
      sfm::ba::solve_cuda(p, o);
    }
    std::vector<std::array<int, 2>> sizes = {
        {12, 1000}, {36, 5000}, {72, 10000}, {144, 20000}, {288, 40000}};
    if (large) {
      sizes.push_back({576, 40000});
      sizes.push_back({1152, 40000});
    }
    for (const auto& sz : sizes) {
      const sfm::ba::Problem base = make(sz[0], sz[1]);
      auto a = base;
      auto t0 = std::chrono::steady_clock::now();
      sfm::ba::Report ra;
      double cpu = 0.0;
      if (!gpu_only) {
        ra = sfm::ba::solve(a, o);
        cpu = ms(t0) / ra.iterations;
      }
      auto b = base;
      t0 = std::chrono::steady_clock::now();
      const auto rb = sfm::ba::solve_cuda(b, o);
      const double gpu = ms(t0) / rb.iterations;
      std::printf("%8d %8d %8zu | %12.2f %12.2f %7.1fx   (final cost cpu %.6g, cuda %.6g)\n", sz[0],
                  sz[1], base.obs.size(), cpu, gpu, cpu / gpu, ra.final_cost, rb.final_cost);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_ba: %s\n", e.what());
    return 1;
  }
}
