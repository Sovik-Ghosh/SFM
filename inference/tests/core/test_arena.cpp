#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

#include "sfm/core/arena.hpp"
#include "sfm/core/cuda_check.h"

using sfm::core::Arena;

namespace {
bool aligned(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }
}  // namespace

TEST(Arena, AllocationsAreAlignedAndCounted) {
  Arena arena(1 << 20);
  auto* a = arena.allocate<char>(3);    // odd size
  auto* b = arena.allocate<float>(10);  // must still start on a 256-B boundary
  auto* c = arena.allocate<double>(1);
  EXPECT_TRUE(aligned(a, 256));
  EXPECT_TRUE(aligned(b, 256));
  EXPECT_TRUE(aligned(c, 256));
  EXPECT_EQ(reinterpret_cast<char*>(b) - a, 256);
  EXPECT_EQ(arena.used(), std::size_t{2} * 256 + sizeof(double));
  EXPECT_EQ(arena.capacity(), 1u << 20);
}

TEST(Arena, LargerAlignmentUsesTheAbsoluteAddress) {
  Arena arena(1 << 20);
  arena.allocate<char>(1);
  void* p = arena.allocate_bytes(16, 4096);
  EXPECT_TRUE(aligned(p, 4096));
}

TEST(Arena, OverflowThrowsWithSizesInMessage) {
  Arena arena(1024);
  arena.allocate<char>(600);
  try {
    arena.allocate<char>(600);  // 600 fits by itself but not after the first (aligned to 768)
    FAIL() << "expected ArenaError";
  } catch (const sfm::ArenaError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("600 B"), std::string::npos) << msg;
    EXPECT_NE(msg.find("1024 B"), std::string::npos) << msg;
  }
  EXPECT_EQ(arena.used(), 600u);  // failed request changes nothing
}

TEST(Arena, ElementCountOverflowIsRejected) {
  Arena arena(1024);
  EXPECT_THROW(arena.allocate<double>(std::numeric_limits<std::size_t>::max() / 4),
               sfm::ArenaError);
}

TEST(Arena, BadAlignmentIsRejected) {
  Arena arena(1024);
  EXPECT_THROW(arena.allocate_bytes(8, 3), sfm::ArenaError);
  EXPECT_THROW(arena.allocate_bytes(8, 0), sfm::ArenaError);
}

TEST(Arena, ResetReusesMemoryAndKeepsHighWater) {
  Arena arena(1 << 16);
  void* first = arena.allocate<int>(100);
  arena.allocate<int>(1000);
  const std::size_t peak = arena.used();
  arena.reset();
  EXPECT_EQ(arena.used(), 0u);
  EXPECT_EQ(arena.high_water(), peak);
  EXPECT_EQ(arena.allocate<int>(1), first);
}

TEST(Arena, ZeroSizedRequestReturnsNull) {
  Arena arena(1024);
  EXPECT_EQ(arena.allocate<int>(0), nullptr);
  EXPECT_EQ(arena.used(), 0u);
}

TEST(Arena, MemoryIsUsableDeviceMemory) {
  Arena arena(1 << 20);
  auto* d = arena.allocate<unsigned char>(4096);
  CHECK(cudaMemset(d, 0xAB, 4096));
  unsigned char h[4096];
  CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  for (unsigned char v : h) ASSERT_EQ(v, 0xAB);
}
