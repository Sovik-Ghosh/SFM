#include <gtest/gtest.h>

#include "sfm/core/cuda_check.h"
#include "sfm/core/cuda_timer.hpp"
#include "sfm/core/error.hpp"
#include "sfm/core/stream.hpp"

#include "test_env.hpp"

using sfm::core::CudaTimer;
using sfm::core::Event;
using sfm::core::Stream;

namespace {

// Busy-waits roughly `cycles` GPU clock cycles.
__global__ void spin(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

float time_spin(const Stream& s, long long cycles) {
  CudaTimer t(s);
  t.start();
  spin<<<1, 1, 0, s.get()>>>(cycles);
  CHECK(cudaGetLastError());
  t.stop();
  return t.elapsed_ms();
}

}  // namespace

TEST(Stream, IsNonBlocking) {
  Stream s;
  unsigned int flags = 0;
  CHECK(cudaStreamGetFlags(s.get(), &flags));
  EXPECT_EQ(flags, static_cast<unsigned int>(cudaStreamNonBlocking));
}

TEST(Stream, MoveTransfersHandle) {
  Stream a;
  cudaStream_t h = a.get();
  Stream b(std::move(a));
  EXPECT_EQ(b.get(), h);
  EXPECT_EQ(a.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(Event, DoneAfterSynchronize) {
  Stream s;
  Event e;
  spin<<<1, 1, 0, s.get()>>>(1000);
  CHECK(cudaGetLastError());
  e.record(s);
  e.synchronize();
  EXPECT_TRUE(e.done());
}

TEST(CudaTimer, LongerWorkTakesLonger) {
  SFM_SKIP_UNDER_SANITIZER();
  Stream s;
  time_spin(s, 1000);  // warm-up
  const float short_ms = time_spin(s, 1'000'000);
  const float long_ms = time_spin(s, 20'000'000);
  EXPECT_GT(short_ms, 0.0F);
  EXPECT_GT(long_ms, 5.0F * short_ms) << short_ms << " vs " << long_ms;
}

TEST(CudaTimer, MisuseThrows) {
  Stream s;
  CudaTimer t(s);
  EXPECT_THROW(t.stop(), sfm::Error);
  EXPECT_THROW(t.elapsed_ms(), sfm::Error);
  t.start();
  EXPECT_THROW(t.elapsed_ms(), sfm::Error);
}
