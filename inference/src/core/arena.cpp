#include "sfm/core/arena.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

#include "sfm/core/cuda_check.h"

namespace sfm::core {

namespace {

bool is_power_of_two(std::size_t x) { return x != 0 && (x & (x - 1)) == 0; }

std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) & ~(a - 1); }

}  // namespace

Arena::Arena(std::size_t capacity_bytes) : capacity_(capacity_bytes) {
  if (capacity_ > 0) CHECK(cudaMalloc(reinterpret_cast<void**>(&base_), capacity_));
}

Arena::~Arena() {
  if (base_ != nullptr) cudaFree(base_);
}

void* Arena::allocate_bytes(std::size_t bytes, std::size_t alignment) {
  if (!is_power_of_two(alignment)) {
    throw ArenaError("Arena: alignment " + std::to_string(alignment) + " is not a power of two");
  }
  if (bytes == 0) return nullptr;

  // Align the absolute address, not the offset: cudaMalloc only guarantees 256-B alignment of
  // base_, so a larger alignment (e.g. 4096) needs the base address taken into account.
  const std::size_t align = std::max(alignment, kAlignment);
  const auto base = reinterpret_cast<std::uintptr_t>(base_);
  const std::size_t offset = align_up(base + used_, align) - base;
  if (offset > capacity_ || bytes > capacity_ - offset) {
    throw ArenaError("Arena: request of " + std::to_string(bytes) + " B does not fit (used " +
                     std::to_string(used_) + " B of " + std::to_string(capacity_) + " B)");
  }
  used_ = offset + bytes;
  high_water_ = std::max(high_water_, used_);
  return base_ + offset;
}

}  // namespace sfm::core
