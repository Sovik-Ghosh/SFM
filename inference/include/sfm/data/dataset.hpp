#pragma once
// The dataset in host memory: gray images and masks in one contiguous pinned block each (a single
// async upload), plus the ground-truth projection matrices and their K, R, t.
// File names follow the Bonn layout, for ids first..last:
// <images_dir>/NNNN.ppm, <masks_dir>/NNNN.pgm, <calib_dir>/NNNN.txt.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/core/pinned_buffer.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/krt.hpp"

#if SFM_WITH_CUDA
#include "sfm/core/arena.hpp"
#include "sfm/core/stream.hpp"
#endif

namespace sfm::data {

struct Dataset {
  std::vector<int> ids;  // image ids, in load order
  int width = 0;         // all images share one size
  int height = 0;
  core::PinnedBuffer<std::uint8_t> gray;   // count() * pixels(); image i starts at i * pixels()
  core::PinnedBuffer<std::uint8_t> masks;  // same layout; empty when use_masks is false
  std::vector<Mat34> P;                    // ground-truth projection per image
  std::vector<Krt> cameras;                // decompose_projection(P[i]): K, R, t per image

  std::size_t count() const { return ids.size(); }
  std::size_t pixels() const {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  }
  const std::uint8_t* gray_image(std::size_t i) const { return gray.data() + i * pixels(); }
  const std::uint8_t* mask_image(std::size_t i) const { return masks.data() + i * pixels(); }
  bool has_masks() const { return !masks.empty(); }
};

// Loads every image, mask and calibration in the configured id range. Throws IoError on a missing
// or unreadable file, an image whose size differs from the first one, or a singular P.
Dataset load_dataset(const config::DatasetConfig& cfg, core::Logger& log);

#if SFM_WITH_CUDA
// Device copies, allocated from the arena. Valid until the arena is reset.
struct DeviceImages {
  std::uint8_t* gray = nullptr;
  std::uint8_t* masks = nullptr;  // nullptr when the dataset has no masks
};

// Enqueues the uploads on `stream` (asynchronous: synchronise before using the data on the host
// side of another stream). Throws ArenaError if the arena is too small.
DeviceImages upload(const Dataset& ds, core::Arena& arena, const core::Stream& stream);
#endif

}  // namespace sfm::data
