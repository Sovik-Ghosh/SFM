#include "sfm/core/cuda_timer.hpp"

#include "sfm/core/cuda_check.h"
#include "sfm/core/error.hpp"

namespace sfm::core {

void CudaTimer::start() {
  start_.record(*stream_);
  started_ = true;
  stopped_ = false;
}

void CudaTimer::stop() {
  if (!started_) throw Error("CudaTimer::stop() called before start()");
  stop_.record(*stream_);
  stopped_ = true;
}

float CudaTimer::elapsed_ms() const {
  if (!started_ || !stopped_) throw Error("CudaTimer::elapsed_ms() needs start() and stop()");
  stop_.synchronize();
  float ms = 0.0F;
  CHECK(cudaEventElapsedTime(&ms, start_.get(), stop_.get()));
  return ms;
}

}  // namespace sfm::core
