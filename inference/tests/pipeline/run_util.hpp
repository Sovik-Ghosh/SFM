#pragma once
// Helpers for the pipeline end-to-end tests.
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "sfm/config/config.hpp"

namespace sfm::pipeline::test {

// Peak resident set of this process so far (GB, Linux VmHWM; 0 if unavailable). ctest runs every
// test case in its own process, so in a gate this is the run's peak plus the test binary.
inline double peak_rss_gb() {
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmHWM:") {
      double kb = 0.0;
      in >> kb;
      return kb / (1024.0 * 1024.0);
    }
    std::string rest;
    std::getline(in, rest);
  }
  return 0.0;
}

inline std::string read_file(const std::filesystem::path& f) {
  std::ifstream in(f);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// configs/<name> with output under the temp dir, no file exports (faster), dumps off.
inline config::Config bunny_config(const std::string& name, const std::string& out_subdir) {
  const std::filesystem::path src(SFM_SOURCE_DIR);
  auto cfg = config::load_config(src / "configs" / name, src);
  cfg.run.output_dir = std::filesystem::temp_directory_path() / out_subdir;
  cfg.output.dump_stages = false;
  cfg.output.ply = false;
  cfg.output.colmap = false;
  std::filesystem::remove_all(cfg.run.output_dir);
  return cfg;
}

}  // namespace sfm::pipeline::test
