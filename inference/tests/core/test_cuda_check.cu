#include <gtest/gtest.h>

#include "sfm/core/cuda_check.h"

namespace {

__global__ void write_index(int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = i;
}

}  // namespace

TEST(CudaCheck, SuccessDoesNotThrow) { EXPECT_NO_THROW(CHECK(cudaSuccess)); }

TEST(CudaCheck, InvalidDeviceThrowsWithCode) {
  try {
    CHECK(cudaSetDevice(9999));
    FAIL() << "expected CudaError";
  } catch (const sfm::core::CudaError& e) {
    EXPECT_EQ(e.code(), cudaErrorInvalidDevice);
    EXPECT_NE(std::string(e.what()).find("cudaSetDevice"), std::string::npos);
  }
}

TEST(CudaCheck, BadLaunchConfigIsCaughtByCheckKernel) {
  write_index<<<1, 4096>>>(nullptr, 0);  // 4096 threads/block exceeds the 1024 limit
  EXPECT_THROW(CHECK_KERNEL(), sfm::core::CudaError);
}

TEST(CudaCheck, KernelRoundTrip) {
  constexpr int n = 1000;
  int* d_out = nullptr;
  CHECK(cudaMalloc(&d_out, n * sizeof(int)));
  write_index<<<(n + 255) / 256, 256>>>(d_out, n);
  CHECK_KERNEL();
  int h_out[n];
  CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_out));
  for (int i = 0; i < n; ++i) ASSERT_EQ(h_out[i], i);
}
