// CPU vs CUDA features (ORB or SIFT): wall time per image (median over runs after warm-up), device
// time from CUDA events, and a parity check of every CUDA result.
//   bench_features [--method orb|sift] [--images N] [--runs R] [--warmup W] [--root data/bunny]
//                  [--no-mask 1]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/data/image_io.hpp"
#include "sfm/features/orb.hpp"
#include "sfm/features/orb_cuda.hpp"
#include "sfm/features/sift.hpp"
#include "sfm/features/sift_cuda.hpp"

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  int images = 36;
  int runs = 20;
  int warmup = 3;
  std::filesystem::path root = "data/bunny";
  std::string method = "orb";
  bool use_mask = true;
  for (int k = 1; k + 1 < argc; k += 2) {
    const std::string a = argv[k];
    if (a == "--images") {
      images = std::atoi(argv[k + 1]);
    } else if (a == "--runs") {
      runs = std::atoi(argv[k + 1]);
    } else if (a == "--warmup") {
      warmup = std::atoi(argv[k + 1]);
    } else if (a == "--root") {
      root = argv[k + 1];
    } else if (a == "--method") {
      method = argv[k + 1];
    } else if (a == "--no-mask") {
      use_mask = std::atoi(argv[k + 1]) == 0;
    }
  }
  try {
    std::vector<sfm::data::Image8> gray;
    std::vector<sfm::data::Image8> mask;
    for (int i = 0; i < images; ++i) {
      char id[16];
      std::snprintf(id, sizeof id, "%04d", i);
      gray.push_back(sfm::data::load_gray(root / "images" / (std::string(id) + ".ppm")));
      if (use_mask) {
        mask.push_back(sfm::data::load_mask(root / "silhouettes" / (std::string(id) + ".pgm")));
      }
    }
    if (method != "orb" && method != "sift") throw std::runtime_error("--method orb|sift");
    const auto m = [&](std::size_t k) { return use_mask ? &mask[k] : nullptr; };
    const sfm::features::OrbParams p;
    const sfm::features::SiftParams sp;
    std::unique_ptr<sfm::features::OrbCuda> orb;
    std::unique_ptr<sfm::features::SiftCuda> sift;
    if (method == "orb") {
      orb = std::make_unique<sfm::features::OrbCuda>(p, gray[0].width, gray[0].height);
    } else {
      sift = std::make_unique<sfm::features::SiftCuda>(sp, gray[0].width, gray[0].height);
    }
    std::vector<sfm::features::Features> ref;
    std::vector<double> cpu_ms;
    std::size_t keypoints = 0;
    for (int i = 0; i < images; ++i) {
      const auto k = static_cast<std::size_t>(i);
      const auto t0 = std::chrono::steady_clock::now();
      ref.push_back(method == "orb" ? sfm::features::detect_orb(gray[k], m(k), p)
                                    : sfm::features::detect_sift(gray[k], m(k), sp));
      cpu_ms.push_back(ms_since(t0));
      keypoints += ref.back().size();
    }
    std::vector<double> wall;
    std::vector<double> device;
    int mismatches = 0;
    for (int r = -warmup; r < runs; ++r) {
      for (int i = 0; i < images; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const auto t0 = std::chrono::steady_clock::now();
        const auto f = orb ? orb->detect(gray[k], m(k)) : sift->detect(gray[k], m(k));
        const double ms = ms_since(t0);
        if (r < 0) continue;
        wall.push_back(ms);
        device.push_back(orb ? orb->last_device_ms() : sift->last_device_ms());
        mismatches += (f.xy != ref[k].xy || f.desc != ref[k].desc ||
                       f.response != ref[k].response || f.angle != ref[k].angle)
                          ? 1
                          : 0;
      }
    }
    const double c = median(cpu_ms);
    const double g = median(wall);
    if (method == "orb") {
      std::printf("images %d (%dx%d), ORB max_features %u, levels %d\n", images, gray[0].width,
                  gray[0].height, p.max_features, p.levels);
    } else {
      std::printf("images %d (%dx%d), SIFT max_features %u, octaves %d (first %d)\n", images,
                  gray[0].width, gray[0].height, sp.max_features, sp.octaves, sp.first_octave);
    }
    std::printf("keypoints      : %.0f / image%s\n", static_cast<double>(keypoints) / images,
                use_mask ? " (masked)" : "");
    std::printf("cpu  (1 thread): median %.2f ms / image (1 run per image)\n", c);
    std::printf("cuda wall      : median %.2f ms / image over %d runs (device %.2f ms)\n", g, runs,
                median(device));
    std::printf("speed-up       : %.1fx\n", c / g);
    std::printf("parity         : %d mismatching results of %zu\n", mismatches, wall.size());
    return mismatches == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_features: %s\n", e.what());
    return 1;
  }
}
