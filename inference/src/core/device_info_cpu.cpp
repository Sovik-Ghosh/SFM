// CPU-only build (SFM_WITH_CUDA=OFF): there are no CUDA devices.
#include <string>

#include "sfm/core/device_info.hpp"
#include "sfm/core/error.hpp"

namespace sfm::core {

int device_count() { return 0; }

DeviceInfo query_device(int index) {
  throw Error("query_device(" + std::to_string(index) +
              "): this build has no CUDA support (SFM_WITH_CUDA=OFF)");
}

bool kernels_runnable(int, std::string& why) {
  why = "built without CUDA";
  return false;
}

}  // namespace sfm::core
