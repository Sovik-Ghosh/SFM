#include "sfm/core/cuda_check.h"
#include "sfm/core/device_info.hpp"

namespace sfm::core {

namespace {
__global__ void probe_kernel(int* p) {
  if (p != nullptr) *p = 1;
}
}  // namespace

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

bool kernels_runnable(int index, std::string& why) {
  int previous = 0;
  if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(index) != cudaSuccess) {
    why = cudaGetErrorString(cudaGetLastError());
    return false;
  }
  // Resolving a kernel's attributes needs its code for this device (SASS, or PTX the driver can
  // JIT): the same check every launch would fail, without launching anything.
  cudaFuncAttributes attr{};
  const cudaError_t e = cudaFuncGetAttributes(&attr, probe_kernel);
  cudaGetLastError();  // not sticky, but clear it for the caller
  cudaSetDevice(previous);
  if (e != cudaSuccess) {
    why = cudaGetErrorString(e);
    return false;
  }
  return true;
}

}  // namespace sfm::core
