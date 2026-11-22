// Loads a run config and its dataset, uploads the images to the GPU, and prints a summary with the
// memory budget. A manual check before a run; P2's sfm_run starts the same way.
//
//   sfm_inspect [config.json]        (default: configs/bunny.json; relative paths from the cwd)
// Exit codes: 0 ok, 1 error (message on stderr), 2 usage.
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "sfm/config/config.hpp"
#include "sfm/core/arena.hpp"
#include "sfm/core/cuda_check.h"
#include "sfm/core/cuda_timer.hpp"
#include "sfm/core/device_info.hpp"
#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/core/stream.hpp"
#include "sfm/data/dataset.hpp"

namespace {

double mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

int run(const std::filesystem::path& config_path) {
  const auto cfg = sfm::config::load_config(config_path);
  sfm::core::Logger log(cfg.run.log_level, &std::cerr);
  log.info("config.loaded", {{"file", config_path.string()}, {"run", cfg.run.name}});

  const int devices = sfm::core::device_count();
  if (cfg.device.index >= devices) {
    throw sfm::ConfigError("device.index: " + std::to_string(cfg.device.index) + " but only " +
                           std::to_string(devices) + " CUDA device(s) visible");
  }
  CHECK(cudaSetDevice(cfg.device.index));
  const auto dev = sfm::core::query_device(cfg.device.index);

  const auto ds = sfm::data::load_dataset(cfg.dataset, log);
  sfm::core::Arena arena(static_cast<std::size_t>(cfg.memory.arena_mb) << 20);
  sfm::core::Stream stream;
  sfm::core::CudaTimer timer(stream);
  timer.start();
  sfm::data::upload(ds, arena, stream);
  timer.stop();
  stream.synchronize();

  const std::size_t image_bytes = ds.gray.bytes() + ds.masks.bytes();
  std::printf("run       %s  (config schema v%d, seed %llu)\n", cfg.run.name.c_str(),
              cfg.schema_version, static_cast<unsigned long long>(cfg.run.seed));
  std::printf("device    [%d] %s  sm_%d%d  %.1f GiB\n", dev.index, dev.name.c_str(), dev.cc_major,
              dev.cc_minor, mib(dev.total_global_mem) / 1024.0);
  std::printf("dataset   %s\n", cfg.dataset.root.string().c_str());
  std::printf("          %zu images (ids %d..%d), %dx%d, masks %s\n", ds.count(), ds.ids.front(),
              ds.ids.back(), ds.width, ds.height, ds.has_masks() ? "yes" : "no");
  std::printf("memory    images on host (pinned) %.1f MiB\n", mib(image_bytes));
  std::printf("          arena %.0f MiB, used %.1f MiB (%.1f %%)\n", mib(arena.capacity()),
              mib(arena.used()),
              100.0 * static_cast<double>(arena.used()) / static_cast<double>(arena.capacity()));
  std::printf("upload    %.2f ms  (%.1f GB/s, one async copy per buffer)\n", timer.elapsed_ms(),
              static_cast<double>(image_bytes) / (timer.elapsed_ms() * 1e6));
  const auto& cam = ds.cameras.front();
  const auto c = cam.centre();
  std::printf("camera %d  fx %.1f fy %.1f cx %.1f cy %.1f px, centre (%.2f, %.2f, %.2f)\n",
              ds.ids.front(), cam.K(0, 0), cam.K(1, 1), cam.K(0, 2), cam.K(1, 2), c.x(), c.y(),
              c.z());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 ||
      (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
    std::fprintf(stderr, "usage: %s [config.json]\n", argv[0]);
    return 2;
  }
  try {
    return run(argc == 2 ? argv[1] : "configs/bunny.json");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
