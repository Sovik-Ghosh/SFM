#pragma once
// DeviceBuffer<T>: owns n elements of device memory (one cudaMalloc). Move-only.
// Copies are asynchronous on a Stream; for true overlap the host side should be a PinnedBuffer.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "sfm/core/cuda_check.h"
#include "sfm/core/stream.hpp"

namespace sfm::core {

template <class T>
class DeviceBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "device memory holds trivially copyable types only");

 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : size_(n) {
    if (n > 0) CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)));
  }
  ~DeviceBuffer() { release(); }

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      release();
      data_ = std::exchange(o.data_, nullptr);
      size_ = std::exchange(o.size_, 0);
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }
  bool empty() const noexcept { return size_ == 0; }

  // Copy n elements host -> device, starting at element `offset`.
  void upload(const T* h_src, std::size_t n, const Stream& s, std::size_t offset = 0) {
    check_range(n, offset);
    CHECK(cudaMemcpyAsync(data_ + offset, h_src, n * sizeof(T), cudaMemcpyHostToDevice, s.get()));
  }

  // Copy n elements device -> host, starting at element `offset`.
  void download(T* h_dst, std::size_t n, const Stream& s, std::size_t offset = 0) const {
    check_range(n, offset);
    CHECK(cudaMemcpyAsync(h_dst, data_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost, s.get()));
  }

 private:
  void check_range(std::size_t n, std::size_t offset) const {
    if (offset > size_ || n > size_ - offset) {
      throw std::out_of_range("DeviceBuffer: [" + std::to_string(offset) + ", " +
                              std::to_string(offset + n) + ") exceeds size " +
                              std::to_string(size_));
    }
  }
  void release() noexcept {
    if (data_ != nullptr) cudaFree(data_);  // destructor path: never throws
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace sfm::core
