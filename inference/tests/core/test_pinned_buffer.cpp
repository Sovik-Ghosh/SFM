#include <gtest/gtest.h>

#include <numeric>

#include "sfm/core/cuda_check.h"
#include "sfm/core/pinned_buffer.hpp"

using sfm::core::PinnedBuffer;

TEST(PinnedBuffer, MemoryIsPageLockedHost) {
  PinnedBuffer<float> buf(1024);
  cudaPointerAttributes attr{};
  CHECK(cudaPointerGetAttributes(&attr, buf.data()));
  EXPECT_EQ(attr.type, cudaMemoryTypeHost);
}

TEST(PinnedBuffer, ElementAccessAndIteration) {
  PinnedBuffer<int> buf(5);
  std::iota(buf.begin(), buf.end(), 10);
  EXPECT_EQ(buf[0], 10);
  EXPECT_EQ(buf[4], 14);
  EXPECT_EQ(std::accumulate(buf.begin(), buf.end(), 0), 60);
  EXPECT_EQ(buf.bytes(), 5 * sizeof(int));
}

TEST(PinnedBuffer, MoveTransfersOwnership) {
  PinnedBuffer<int> a(8);
  int* p = a.data();
  PinnedBuffer<int> b(std::move(a));
  EXPECT_EQ(b.data(), p);
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move)
}
