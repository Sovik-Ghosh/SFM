#include "sfm/data/dataset.hpp"

#include <cstring>
#include <string>

#if SFM_WITH_CUDA
#include "sfm/core/cuda_check.h"
#endif
#include "sfm/core/error.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/krt.hpp"

namespace sfm::data {

namespace {

// 7, "ppm" -> "0007.ppm" (ids are 0..9999, checked by the caller)
std::string file_name(int id, const char* ext) {
  std::string s = std::to_string(id);
  s.insert(0, s.size() < 4 ? 4 - s.size() : 0, '0');
  return s + "." + ext;
}

// Copies one loaded image into slot i of a contiguous buffer, checking its size.
void place(const Image8& img, const std::filesystem::path& path, Dataset& ds,
           core::PinnedBuffer<std::uint8_t>& dst, std::size_t i) {
  if (img.width != ds.width || img.height != ds.height) {
    throw IoError("dataset " + path.string() + ": size " + std::to_string(img.width) + "x" +
                  std::to_string(img.height) + " differs from " + std::to_string(ds.width) + "x" +
                  std::to_string(ds.height));
  }
  std::memcpy(dst.data() + i * ds.pixels(), img.pixels.data(), ds.pixels());
}

}  // namespace

Dataset load_dataset(const config::DatasetConfig& cfg, core::Logger& log) {
  if (cfg.first < 0 || cfg.last < cfg.first || cfg.last > 9999) {
    throw IoError("dataset: invalid id range " + std::to_string(cfg.first) + ".." +
                  std::to_string(cfg.last));
  }
  const auto n = static_cast<std::size_t>(cfg.image_count());
  const auto images = cfg.root / cfg.images_dir;
  const auto masks = cfg.root / cfg.masks_dir;
  const auto calib = cfg.root / cfg.calib_dir;

  Dataset ds;
  for (int id = cfg.first; id <= cfg.last; ++id) {
    const std::size_t i = ds.ids.size();
    const auto image_path = images / file_name(id, "ppm");
    const Image8 gray = load_gray(image_path);
    if (i == 0) {  // first image fixes the size and the buffer layout
      ds.width = gray.width;
      ds.height = gray.height;
      ds.gray = core::PinnedBuffer<std::uint8_t>(n * ds.pixels());
      if (cfg.use_masks) ds.masks = core::PinnedBuffer<std::uint8_t>(n * ds.pixels());
    }
    place(gray, image_path, ds, ds.gray, i);
    if (cfg.use_masks) {
      const auto mask_path = masks / file_name(id, "pgm");
      place(load_mask(mask_path), mask_path, ds, ds.masks, i);
    }
    ds.P.push_back(read_projection(calib / file_name(id, "txt")));
    ds.cameras.push_back(decompose_projection(ds.P.back()));
    ds.ids.push_back(id);
  }

  log.info("dataset.loaded", {{"root", cfg.root.string()},
                              {"images", ds.count()},
                              {"width", ds.width},
                              {"height", ds.height},
                              {"masks", ds.has_masks()},
                              {"host_mib", static_cast<double>(ds.gray.bytes() + ds.masks.bytes()) /
                                               (1024.0 * 1024.0)}});
  return ds;
}

#if SFM_WITH_CUDA
DeviceImages upload(const Dataset& ds, core::Arena& arena, const core::Stream& stream) {
  DeviceImages d;
  d.gray = arena.allocate<std::uint8_t>(ds.gray.size());
  CHECK(cudaMemcpyAsync(d.gray, ds.gray.data(), ds.gray.bytes(), cudaMemcpyHostToDevice,
                        stream.get()));
  if (ds.has_masks()) {
    d.masks = arena.allocate<std::uint8_t>(ds.masks.size());
    CHECK(cudaMemcpyAsync(d.masks, ds.masks.data(), ds.masks.bytes(), cudaMemcpyHostToDevice,
                          stream.get()));
  }
  return d;
}
#endif

}  // namespace sfm::data
