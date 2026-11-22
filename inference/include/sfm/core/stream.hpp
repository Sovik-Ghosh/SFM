#pragma once
// RAII wrappers for CUDA streams and events. Move-only; destroyed handles are released without
// throwing.

#include <cuda_runtime.h>

namespace sfm::core {

// A non-blocking stream (does not synchronise with the legacy default stream).
class Stream {
 public:
  Stream();
  ~Stream();
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  cudaStream_t get() const noexcept { return stream_; }
  void synchronize() const;  // blocks the host until all work on this stream is done

 private:
  cudaStream_t stream_ = nullptr;
};

class Event {
 public:
  Event();  // timing enabled
  ~Event();
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  cudaEvent_t get() const noexcept { return event_; }
  void record(const Stream& stream);
  void synchronize() const;
  bool done() const;  // true once all work captured by record() has finished

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace sfm::core
