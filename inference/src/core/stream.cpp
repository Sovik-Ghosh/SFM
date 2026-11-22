#include "sfm/core/stream.hpp"

#include <utility>

#include "sfm/core/cuda_check.h"

namespace sfm::core {

Stream::Stream() { CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

Stream::~Stream() {
  if (stream_ != nullptr) cudaStreamDestroy(stream_);  // destructor must not throw
}

Stream::Stream(Stream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

void Stream::synchronize() const { CHECK(cudaStreamSynchronize(stream_)); }

Event::Event() { CHECK(cudaEventCreate(&event_)); }

Event::~Event() {
  if (event_ != nullptr) cudaEventDestroy(event_);
}

Event::Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) cudaEventDestroy(event_);
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

void Event::record(const Stream& stream) { CHECK(cudaEventRecord(event_, stream.get())); }

void Event::synchronize() const { CHECK(cudaEventSynchronize(event_)); }

bool Event::done() const {
  const cudaError_t status = cudaEventQuery(event_);
  if (status == cudaErrorNotReady) return false;
  CHECK(status);
  return true;
}

}  // namespace sfm::core
