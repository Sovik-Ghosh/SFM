#include "sfm/core/cuda_check.h"
#include "sfm/core/device_info.hpp"

namespace sfm::core {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error: "no GPU" is a valid answer here
    return 0;
  }
  return count;
}

DeviceInfo query_device(int index) {
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, index));

  DeviceInfo info;
  info.index = index;
  info.name = prop.name;
  info.cc_major = prop.major;
  info.cc_minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info.shared_mem_per_block = prop.sharedMemPerBlock;
  info.total_global_mem = prop.totalGlobalMem;
  return info;
}

}  // namespace sfm::core
