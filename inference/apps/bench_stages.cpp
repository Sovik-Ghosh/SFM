// CPU vs GPU per stage on one config: every stage on the CPU methods, then on the CUDA methods
// (sfm::pipeline::use_backend), each the best of --repeat runs after one warm-up run. Prints wall
// seconds per stage, the speed-up and the result of each backend.
//   bench_stages <config.json> [--repeat N] [--backend both|cpu|gpu]
// CPU-only builds and machines without a GPU measure the CPU column only.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/registry.hpp"

namespace fs = std::filesystem;

namespace {

struct Column {
  bool ran = false;
  std::string methods;
  sfm::pipeline::Timings best;  // per stage: the fastest run
  int registered = 0;
  int images = 0;
  int points = 0;
  double rotation_median_deg = -1.0;
};

Column measure(const sfm::config::Config& base, bool gpu, int repeat) {
  sfm::config::Config cfg = sfm::pipeline::use_backend(base, gpu);
  cfg.run.output_dir = fs::temp_directory_path() / "sfm_bench_stages";
  cfg.run.name += gpu ? "-gpu" : "-cpu";
  cfg.run.log_level = sfm::core::LogLevel::kError;
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  Column c;
  c.methods = cfg.features.method + " / " + cfg.matching.method + " / " + cfg.ba.method;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  c.best = {kInf, kInf, kInf, kInf, kInf, kInf, kInf, kInf};
  std::ostringstream discard;
  sfm::core::Logger log(cfg.run.log_level, &discard, false);
  for (int r = 0; r <= repeat; ++r) {  // run 0 warms up (CUDA context, first allocations)
    const auto res = sfm::pipeline::run_pipeline(cfg, {}, log);
    if (r == 0) {
      if (res.metrics) {
        c.registered = static_cast<int>(res.metrics->registered_ids.size());
        c.images = res.metrics->images;
        c.points = res.metrics->points;
        c.rotation_median_deg = res.metrics->rotation_median_deg;
      }
      continue;
    }
    const auto& t = res.timings;
    auto& b = c.best;
    b.load = std::min(b.load, t.load);
    b.features = std::min(b.features, t.features);
    b.matching = std::min(b.matching, t.matching);
    b.tracks = std::min(b.tracks, t.tracks);
    b.reconstruction = std::min(b.reconstruction, t.reconstruction);
    b.ba = std::min(b.ba, t.ba);
    b.export_ = std::min(b.export_, t.export_);
    b.total = std::min(b.total, t.total);
  }
  fs::remove_all(cfg.run.output_dir);
  c.ran = true;
  return c;
}

int usage() {
  std::fprintf(stderr, "usage: bench_stages <config.json> [--repeat N] [--backend both|cpu|gpu]\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  int repeat = 3;
  std::string which = "both";
  for (int k = 2; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--repeat" && k + 1 < argc) {
      repeat = std::max(1, std::stoi(argv[++k]));
    } else if (a == "--backend" && k + 1 < argc) {
      which = argv[++k];
    } else {
      return usage();
    }
  }
  try {
    const auto cfg = sfm::config::load_config(argv[1]);
    const auto backend = sfm::pipeline::detect_backend();
    const bool gpu_ok = sfm::pipeline::built_with_cuda() && backend.gpu;
    Column cpu;
    Column gpu;
    if (which != "gpu") cpu = measure(cfg, false, repeat);
    if (which != "cpu") {
      if (gpu_ok) {
        gpu = measure(cfg, true, repeat);
      } else {
        std::printf("GPU column skipped: %s\n", sfm::pipeline::built_with_cuda()
                                                    ? "no CUDA device visible"
                                                    : "built without CUDA (SFM_WITH_CUDA=OFF)");
      }
    }
    std::printf("config %s, best of %d runs after a warm-up%s%s\n", argv[1], repeat,
                gpu_ok ? ", GPU: " : "", gpu_ok ? backend.device.c_str() : "");
    if (cpu.ran) std::printf("CPU methods: %s\n", cpu.methods.c_str());
    if (gpu.ran) std::printf("GPU methods: %s\n", gpu.methods.c_str());
    std::printf("\n%-22s %10s %10s %9s\n", "stage (s)", "CPU", "GPU", "speed-up");
    const auto row = [&](const char* name, double c, double g) {
      std::printf("%-22s", name);
      if (cpu.ran) {
        std::printf(" %10.3f", c);
      } else {
        std::printf(" %10s", "-");
      }
      if (gpu.ran) {
        std::printf(" %10.3f", g);
      } else {
        std::printf(" %10s", "-");
      }
      if (cpu.ran && gpu.ran && g > 0) {
        std::printf(" %8.1fx\n", c / g);
      } else {
        std::printf(" %9s\n", "-");
      }
    };
    row("load", cpu.best.load, gpu.best.load);
    row("features", cpu.best.features, gpu.best.features);
    row("matching", cpu.best.matching, gpu.best.matching);
    row("tracks", cpu.best.tracks, gpu.best.tracks);
    row("reconstruction", cpu.best.reconstruction, gpu.best.reconstruction);
    row("  of which BA", cpu.best.ba, gpu.best.ba);
    row("export", cpu.best.export_, gpu.best.export_);
    row("total", cpu.best.total, gpu.best.total);
    std::printf("\n");
    for (const Column* c : {&cpu, &gpu}) {
      if (!c->ran) continue;
      std::printf("%s: registered %d/%d, points %d, rotation median %s\n",
                  c == &cpu ? "CPU" : "GPU", c->registered, c->images, c->points,
                  c->rotation_median_deg >= 0
                      ? (std::to_string(c->rotation_median_deg).substr(0, 5) + " deg").c_str()
                      : "n/a (no ground truth)");
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_stages: %s\n", e.what());
    return 1;
  }
}
