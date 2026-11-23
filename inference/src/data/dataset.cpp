#include "sfm/data/dataset.hpp"

#include <cstring>
#include <string>

#if SFM_WITH_CUDA
#include "sfm/core/cuda_check.h"
#endif
#include "sfm/core/error.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/image_list.hpp"
#include "sfm/data/krt.hpp"

namespace sfm::data {

namespace {

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
  const auto files = list_images(cfg);
  const std::size_t n = files.size();

  Dataset ds;
  for (const ImageEntry& e : files) {
    const std::size_t i = ds.ids.size();
    const Image8 gray = load_gray(e.image);
    if (i == 0) {  // first image fixes the size and the buffer layout
      ds.width = gray.width;
      ds.height = gray.height;
      ds.gray = core::PinnedBuffer<std::uint8_t>(n * ds.pixels());
      if (cfg.use_masks) ds.masks = core::PinnedBuffer<std::uint8_t>(n * ds.pixels());
    }
    place(gray, e.image, ds, ds.gray, i);
    if (cfg.use_masks) place(load_mask(e.mask, cfg.mask_object_white), e.mask, ds, ds.masks, i);
    if (e.calib.empty()) {  // dataset.camera: P = K [I | 0]
      ds.cameras.push_back(load_camera(cfg, e, gray.width, gray.height));
      ds.P.push_back(compose_projection(ds.cameras.back()));
    } else {  // the file's P as written (any scale)
      ds.P.push_back(read_projection(e.calib));
      ds.cameras.push_back(decompose_projection(ds.P.back()));
    }
    ds.ids.push_back(e.id);
    ds.names.push_back(e.name);
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
