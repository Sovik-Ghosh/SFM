#pragma once
// CUDA error checking. Every CUDA API call is wrapped in CHECK(...).
// Errors are thrown as sfm::core::CudaError so callers of the library decide how to react
// (the CLI app prints and exits; tests assert on the exception).

#include <string>

#include <cuda_runtime.h>

#include "sfm/core/error.hpp"

namespace sfm::core {

class CudaError : public ::sfm::Error {
 public:
  CudaError(cudaError_t code, const char* expr, const char* file, int line)
      : ::sfm::Error(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                     " failed: " + cudaGetErrorName(code) + " (" + cudaGetErrorString(code) + ")"),
        code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

inline void check_cuda(cudaError_t code, const char* expr, const char* file, int line) {
  if (code != cudaSuccess) {
    throw CudaError(code, expr, file, line);
  }
}

}  // namespace sfm::core

#define CHECK(call) ::sfm::core::check_cuda((call), #call, __FILE__, __LINE__)

// After a kernel launch while developing. Never inside timed loops (it synchronises).
#define CHECK_KERNEL()              \
  do {                              \
    CHECK(cudaGetLastError());      \
    CHECK(cudaDeviceSynchronize()); \
  } while (0)
