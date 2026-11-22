// Prints the library version and visible GPUs. Used as the release image smoke check.
#include <cstdio>
#include <exception>

#include "sfm/core/device_info.hpp"
#include "sfm/core/version.hpp"

int main() {
  try {
    std::printf("sfm %s\n", sfm::core::version_string());
    const int n = sfm::core::device_count();
    std::printf("cuda devices: %d\n", n);
    for (int i = 0; i < n; ++i) {
      const auto d = sfm::core::query_device(i);
      std::printf("  [%d] %s  sm_%d%d  %d SMs  %.1f GiB\n", d.index, d.name.c_str(), d.cc_major,
                  d.cc_minor, d.sm_count,
                  static_cast<double>(d.total_global_mem) / (1024.0 * 1024.0 * 1024.0));
    }
    return n > 0 ? 0 : 2;  // exit 2 = no GPU visible (e.g. container started without a GPU device)
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
