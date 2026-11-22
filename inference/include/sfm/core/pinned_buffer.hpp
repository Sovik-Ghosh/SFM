#pragma once
// PinnedBuffer<T>: owns n elements of page-locked host memory (cudaMallocHost). Move-only.
// Page-locked memory lets cudaMemcpyAsync overlap with kernels and reach full PCIe bandwidth.

#include <cstddef>
#include <type_traits>
#include <utility>

#include "sfm/core/cuda_check.h"

namespace sfm::core {

template <class T>
class PinnedBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "pinned memory holds trivially copyable types only");

 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t n) : size_(n) {
    if (n > 0) CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), n * sizeof(T)));
  }
  ~PinnedBuffer() { release(); }

  PinnedBuffer(PinnedBuffer&& o) noexcept
      : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      release();
      data_ = std::exchange(o.data_, nullptr);
      size_ = std::exchange(o.size_, 0);
    }
    return *this;
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }
  bool empty() const noexcept { return size_ == 0; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }
  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) cudaFreeHost(data_);
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace sfm::core
