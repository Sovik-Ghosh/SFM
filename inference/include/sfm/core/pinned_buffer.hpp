#pragma once
// PinnedBuffer<T>: owns n elements of page-locked host memory (cudaMallocHost). Move-only.
// Page-locked memory lets cudaMemcpyAsync overlap with kernels and reach full PCIe bandwidth.
// Without CUDA (SFM_WITH_CUDA=0), or when cudaMallocHost fails (no driver: a CPU-only machine
// running a CUDA build), it is ordinary host memory: the CPU stages only need the bytes.

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

#if SFM_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace sfm::core {

template <class T>
class PinnedBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "pinned memory holds trivially copyable types only");

 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(std::size_t n) : size_(n) {
    if (n == 0) return;
#if SFM_WITH_CUDA
    if (cudaMallocHost(reinterpret_cast<void**>(&data_), n * sizeof(T)) == cudaSuccess) {
      pinned_ = true;
      return;
    }
    cudaGetLastError();  // clear the error: fall back to ordinary memory
    data_ = nullptr;
#endif
    data_ = static_cast<T*>(::operator new(n * sizeof(T)));
  }
  ~PinnedBuffer() { release(); }

  PinnedBuffer(PinnedBuffer&& o) noexcept
      : data_(std::exchange(o.data_, nullptr)),
        size_(std::exchange(o.size_, 0)),
        pinned_(std::exchange(o.pinned_, false)) {}
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) {
      release();
      data_ = std::exchange(o.data_, nullptr);
      size_ = std::exchange(o.size_, 0);
      pinned_ = std::exchange(o.pinned_, false);
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
  // True when the memory is page-locked (cudaMallocHost succeeded).
  bool pinned() const noexcept { return pinned_; }

  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }
  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + size_; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + size_; }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
#if SFM_WITH_CUDA
      if (pinned_) {
        cudaFreeHost(data_);
      } else {
        ::operator delete(data_);
      }
#else
      ::operator delete(data_);
#endif
    }
    data_ = nullptr;
    size_ = 0;
    pinned_ = false;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
  bool pinned_ = false;
};

}  // namespace sfm::core
