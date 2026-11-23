#pragma once

#include <cstddef>
#include <string>

namespace sfm::core {

struct DeviceInfo {
  int index = 0;
  std::string name;
  int cc_major = 0;  // compute capability, e.g. 12.0 -> sm_120
  int cc_minor = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  std::size_t shared_mem_per_block = 0;  // bytes
  std::size_t total_global_mem = 0;      // bytes
};

// Number of visible CUDA devices (0 if no driver/GPU; never throws for that case).
int device_count();

// Properties of one device. Throws CudaError for an invalid index.
DeviceInfo query_device(int index);

// Whether this build's kernels can run on the device: false when the binary has no code for its
// architecture (built with CMAKE_CUDA_ARCHITECTURES that leave the GPU out). `why` gets the CUDA
// error. Never throws.
bool kernels_runnable(int index, std::string& why);

}  // namespace sfm::core
