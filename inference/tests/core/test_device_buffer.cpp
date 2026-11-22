#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"

#include "test_env.hpp"

using sfm::core::DeviceBuffer;
using sfm::core::Stream;

namespace {
std::size_t free_bytes() {
  std::size_t free = 0, total = 0;
  CHECK(cudaMemGetInfo(&free, &total));
  return free;
}
}  // namespace

TEST(DeviceBuffer, AllocationAndReleaseRestoreFreeMemory) {
  SFM_SKIP_UNDER_SANITIZER();
  CHECK(cudaFree(nullptr));  // create the context before measuring
  const std::size_t before = free_bytes();
  {
    DeviceBuffer<float> buf(64u << 20);  // 256 MiB
    EXPECT_EQ(buf.bytes(), 256u << 20);
    EXPECT_LE(free_bytes() + (256u << 20), before + (2u << 20));
  }
  // 2 MiB slack: other processes (desktop compositor) may allocate concurrently.
  EXPECT_NEAR(static_cast<double>(free_bytes()), static_cast<double>(before), 2.0 * (1 << 20));
}

TEST(DeviceBuffer, UploadDownloadRoundTrip) {
  Stream s;
  std::vector<int> h(1000);
  std::iota(h.begin(), h.end(), -500);
  DeviceBuffer<int> d(h.size());
  d.upload(h.data(), h.size(), s);
  std::vector<int> back(h.size(), 0);
  d.download(back.data(), back.size(), s);
  s.synchronize();
  EXPECT_EQ(back, h);
}

TEST(DeviceBuffer, OffsetCopiesTouchOnlyTheirRange) {
  Stream s;
  DeviceBuffer<int> d(10);
  const std::vector<int> zeros(10, 0), three{7, 8, 9};
  d.upload(zeros.data(), 10, s);
  d.upload(three.data(), 3, s, 4);
  std::vector<int> back(10, -1);
  d.download(back.data(), 10, s);
  s.synchronize();
  EXPECT_EQ(back, (std::vector<int>{0, 0, 0, 0, 7, 8, 9, 0, 0, 0}));
}

TEST(DeviceBuffer, OutOfRangeThrowsBeforeTouchingTheGpu) {
  Stream s;
  DeviceBuffer<int> d(4);
  std::vector<int> h(8);
  EXPECT_THROW(d.upload(h.data(), 5, s), std::out_of_range);
  EXPECT_THROW(d.upload(h.data(), 1, s, 4), std::out_of_range);
  EXPECT_THROW(d.download(h.data(), 2, s, 3), std::out_of_range);
  EXPECT_THROW(d.upload(h.data(), 2, s, static_cast<std::size_t>(-1)), std::out_of_range);
  EXPECT_NO_THROW(d.upload(h.data(), 0, s, 4));  // empty copy at the end is fine
}

TEST(DeviceBuffer, MoveTransfersOwnership) {
  DeviceBuffer<double> a(16);
  double* p = a.data();
  DeviceBuffer<double> b(std::move(a));
  EXPECT_EQ(b.data(), p);
  EXPECT_EQ(b.size(), 16u);
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move): moved-from state is specified
  EXPECT_EQ(a.data(), nullptr);
  DeviceBuffer<double> c(4);
  c = std::move(b);  // old c freed, takes b's block
  EXPECT_EQ(c.data(), p);
}

TEST(DeviceBuffer, ZeroSizeHasNoAllocation) {
  DeviceBuffer<int> d(0);
  EXPECT_TRUE(d.empty());
  EXPECT_EQ(d.data(), nullptr);
}
