#include "sfm/data/image_io.hpp"

#include <memory>
#include <string>

#include "sfm/core/error.hpp"

// The one translation unit that compiles stb_image.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNM
#define STBI_ONLY_PNG
#define STBI_FAILURE_USERMSG
#include <stb_image.h>

namespace sfm::data {

namespace {

struct StbiFree {
  void operator()(unsigned char* p) const noexcept { stbi_image_free(p); }
};
using StbiPixels = std::unique_ptr<unsigned char, StbiFree>;

// Loads with the requested channel count (1 or 3); stb converts if the file differs.
StbiPixels load(const std::filesystem::path& path, int channels, int& width, int& height) {
  int in_file = 0;
  StbiPixels px(stbi_load(path.string().c_str(), &width, &height, &in_file, channels));
  if (!px) throw IoError("image " + path.string() + ": " + stbi_failure_reason());
  return px;
}

}  // namespace

Image8 load_gray(const std::filesystem::path& path) {
  Image8 img;
  int channels = 0;
  if (stbi_info(path.string().c_str(), &img.width, &img.height, &channels) == 0) {
    throw IoError("image " + path.string() + ": " + stbi_failure_reason());
  }
  const int want = channels <= 2 ? 1 : 3;  // gray(+alpha) stays gray; RGB(A) is converted
  const StbiPixels px = load(path, want, img.width, img.height);
  const std::size_t n = static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
  img.pixels.resize(n);
  const unsigned char* src = px.get();
  if (want == 1) {
    img.pixels.assign(src, src + n);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      img.pixels[i] = rgb_to_gray(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
    }
  }
  return img;
}

Image8 load_mask(const std::filesystem::path& path) {
  Image8 img;
  const StbiPixels px = load(path, 1, img.width, img.height);
  const std::size_t n = static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
  img.pixels.resize(n);
  for (std::size_t i = 0; i < n; ++i) img.pixels[i] = px.get()[i] == 0 ? 255 : 0;
  return img;
}

}  // namespace sfm::data
