// Runs the SfM pipeline described by a config.
//   sfm_run <config.json> [--from <stage>] [--to <stage>]
// Stages: features, matching, tracks, reconstruction, export. --from loads the earlier stages'
// outputs from <output_dir>/<name>/stages/ (written by a previous run). Relative paths in the
// config are taken from the current directory. Exit code 0 on success, 1 on any error.
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/pipeline.hpp"

namespace {

int usage() {
  std::fprintf(stderr, "usage: sfm_run <config.json> [--from <stage>] [--to <stage>]\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  try {
    sfm::pipeline::RunOptions opt;
    for (int k = 2; k < argc; ++k) {
      const std::string a = argv[k];
      if ((a == "--from" || a == "--to") && k + 1 < argc) {
        (a == "--from" ? opt.from : opt.to) = sfm::pipeline::parse_stage(argv[++k]);
      } else {
        return usage();
      }
    }
    const auto cfg = sfm::config::load_config(argv[1]);
    sfm::core::Logger log(cfg.run.log_level, &std::cerr);
    const auto r = sfm::pipeline::run_pipeline(cfg, opt, log);
    std::printf("run directory: %s\n", r.run_dir.string().c_str());
    if (r.metrics) {
      const auto& m = *r.metrics;
      std::printf("registered %zu/%d, points %d, reprojection median %.3f px",
                  m.registered_ids.size(), m.images, m.points, m.reproj_median_px);
      if (m.rotation_median_deg >= 0.0) {
        std::printf(", rotation error median %.2f deg\n", m.rotation_median_deg);
      } else {
        std::printf(" (no ground-truth poses: no pose errors)\n");
      }
    }
    std::printf("total %.2f s (features %.2f, matching %.2f, reconstruction %.2f incl. BA %.2f)\n",
                r.timings.total, r.timings.features, r.timings.matching, r.timings.reconstruction,
                r.timings.ba);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sfm_run: %s\n", e.what());
    return 1;
  }
}
