// CPU vs GPU pair verification (top-2 + ratio + F-RANSAC over all pairs) on the bunny features.
//   bench_matching [--runs R] [--root data/bunny]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/data/image_io.hpp"
#include "sfm/features/orb_cuda.hpp"
#include "sfm/matching/matching_cuda.hpp"

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  int runs = 5;
  std::filesystem::path root = "data/bunny";
  for (int k = 1; k + 1 < argc; k += 2) {
    const std::string a = argv[k];
    if (a == "--runs") {
      runs = std::atoi(argv[k + 1]);
    } else if (a == "--root") {
      root = argv[k + 1];
    }
  }
  try {
    std::vector<sfm::features::Features> f;
    sfm::features::OrbCuda orb(sfm::features::OrbParams{}, 1024, 768);
    for (int i = 0; i < 36; ++i) {
      char id[16];
      std::snprintf(id, sizeof id, "%04d", i);
      const auto g = sfm::data::load_gray(root / "images" / (std::string(id) + ".ppm"));
      const auto m = sfm::data::load_mask(root / "silhouettes" / (std::string(id) + ".pgm"));
      f.push_back(orb.detect(g, &m));
    }
    sfm::matching::VerifyParams p;
    p.seed = 42;
    auto t0 = std::chrono::steady_clock::now();
    const auto cpu = sfm::matching::verify_all_pairs(f, p);
    const double cpu_ms = ms_since(t0);
    std::vector<double> gpu_ms;
    std::size_t gpu_pairs = 0;
    for (int r = -1; r < runs; ++r) {  // first run = warm-up (context, allocations)
      t0 = std::chrono::steady_clock::now();
      const auto gpu = sfm::matching::verify_all_pairs_cuda(f, p);
      if (r >= 0) gpu_ms.push_back(ms_since(t0));
      gpu_pairs = gpu.size();
    }
    std::sort(gpu_ms.begin(), gpu_ms.end());
    const double g = gpu_ms[gpu_ms.size() / 2];
    std::printf("630 pairs, %zu keypoints / image on average\n", [&] {
      std::size_t n = 0;
      for (const auto& x : f) n += x.size();
      return n / f.size();
    }());
    std::printf("cpu  (1 thread): %.0f ms, %zu verified pairs\n", cpu_ms, cpu.size());
    std::printf("cuda           : median %.1f ms over %d runs, %zu verified pairs\n", g, runs,
                gpu_pairs);
    std::printf("speed-up       : %.0fx\n", cpu_ms / g);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_matching: %s\n", e.what());
    return 1;
  }
}
