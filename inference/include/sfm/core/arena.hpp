#pragma once
// Arena: one cudaMalloc at startup, then bump-pointer sub-allocation with no further cudaMalloc in
// the hot path. All allocations are freed together by reset(). Not thread-safe; not movable
// (pointers handed out refer into the block).

#include <cstddef>
#include <limits>

#include "sfm/core/error.hpp"

namespace sfm::core {

class Arena {
 public:
  static constexpr std::size_t kAlignment = 256;  // cudaMalloc's own alignment; good for coalescing

  explicit Arena(std::size_t capacity_bytes);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  Arena(Arena&&) = delete;
  Arena& operator=(Arena&&) = delete;

  // n elements of T, aligned to max(kAlignment, alignof(T)). n == 0 returns nullptr.
  // Throws ArenaError if the arena is full or n * sizeof(T) overflows.
  template <class T>
  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw ArenaError("Arena: element count overflows size_t");
    }
    return static_cast<T*>(allocate_bytes(n * sizeof(T), alignof(T)));
  }

  void* allocate_bytes(std::size_t bytes, std::size_t alignment = kAlignment);

  void reset() noexcept { used_ = 0; }  // invalidates every pointer handed out
  std::size_t used() const noexcept { return used_; }
  std::size_t high_water() const noexcept { return high_water_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  char* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  std::size_t high_water_ = 0;
};

}  // namespace sfm::core
