#pragma once
// GPU-side timing of work on one stream with a pair of CUDA events.
// Usage: timer.start(); launch(...); timer.stop(); float ms = timer.elapsed_ms();

#include "sfm/core/stream.hpp"

namespace sfm::core {

class CudaTimer {
 public:
  explicit CudaTimer(const Stream& stream) : stream_(&stream) {}

  void start();
  void stop();
  // Milliseconds between start() and stop(). Waits for the stop event. Throws sfm::Error if
  // start() and stop() were not both called.
  float elapsed_ms() const;

 private:
  const Stream* stream_;
  Event start_;
  Event stop_;
  bool started_ = false;
  bool stopped_ = false;
};

}  // namespace sfm::core
