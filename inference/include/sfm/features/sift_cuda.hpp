#pragma once
// SIFT on the GPU (P7). Output is byte-identical to detect_sift (sfm/features/sift.hpp) for the
// same inputs: the device runs the per-pixel work (upsample, blur, DoG, extrema + refinement) and
// the per-keypoint work (orientations, descriptors) with the shared arithmetic of sift_core.hpp;
// the host restores the CPU's detection order, applies the mask and selects the strongest
// locations with the same functions as the CPU (sift_detail).

#include <memory>

#include "sfm/data/image_io.hpp"
#include "sfm/features/features.hpp"
#include "sfm/features/sift.hpp"

namespace sfm::features {

class SiftCuda {
 public:
  // Allocates device buffers for images of width x height (reused by every detect call).
  // Throws CudaError on allocation failure, std::invalid_argument for bad parameters.
  SiftCuda(const SiftParams& params, int width, int height);
  ~SiftCuda();
  SiftCuda(SiftCuda&&) noexcept;
  SiftCuda& operator=(SiftCuda&&) noexcept;
  SiftCuda(const SiftCuda&) = delete;
  SiftCuda& operator=(const SiftCuda&) = delete;

  // Same contract as detect_sift; gray (and mask) must have the size given to the constructor.
  Features detect(const data::Image8& gray, const data::Image8* mask);

  const SiftParams& params() const;
  int width() const;
  int height() const;
  // Device time of the last detect call, from upload to the last download (CUDA events).
  float last_device_ms() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sfm::features
