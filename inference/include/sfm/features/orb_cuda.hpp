#pragma once
// ORB on the GPU (P3). Output is bit-identical to detect_orb (sfm/features/orb.hpp) for the same
// inputs: every step uses the shared integer / correctly-rounded arithmetic of exact.hpp, the
// device does the per-pixel work (pyramid, blur, FAST, NMS + compaction, first selection by FAST
// score, Harris, moments, descriptors) and the host the small per-level steps (Harris ranking of
// <= 2 x quota candidates, level-0 coordinates, angles from the moments).

#include <memory>

#include "sfm/data/image_io.hpp"
#include "sfm/features/features.hpp"
#include "sfm/features/orb.hpp"

namespace sfm::features {

class OrbCuda {
 public:
  // Allocates device buffers for images of width x height (reused by every detect call).
  // Throws CudaError on allocation failure, std::invalid_argument for bad parameters.
  OrbCuda(const OrbParams& params, int width, int height);
  ~OrbCuda();
  OrbCuda(OrbCuda&&) noexcept;
  OrbCuda& operator=(OrbCuda&&) noexcept;
  OrbCuda(const OrbCuda&) = delete;
  OrbCuda& operator=(const OrbCuda&) = delete;

  // Same contract as detect_orb; gray (and mask) must have the size given to the constructor.
  Features detect(const data::Image8& gray, const data::Image8* mask);

  const OrbParams& params() const;
  int width() const;
  int height() const;

  // Device time of the last detect call, from upload to the last download (CUDA events).
  float last_device_ms() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sfm::features
