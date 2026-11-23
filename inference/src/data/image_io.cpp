#include "sfm/data/image_io.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/data/image_list.hpp"

// The one translation unit that compiles stb_image.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNM
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
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

// Pixels of a w x h image with `c` channels as displayed under EXIF orientation o (1..8): the
// stored image rotated / mirrored. Orientations 5-8 swap width and height.
std::vector<unsigned char> orient(const unsigned char* src, int& w, int& h, int c, int o) {
  const int W = w;
  const int H = h;
  const bool swap = o >= 5;
  const int ow = swap ? H : W;
  const int oh = swap ? W : H;
  std::vector<unsigned char> out(static_cast<std::size_t>(ow) * oh * c);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int dx = x;
      int dy = y;
      // 2 mirror left-right, 3 rotate 180, 4 mirror top-bottom, 5 transpose, 6 rotate 90
      // clockwise, 7 transverse, 8 rotate 90 counter-clockwise.
      if (o == 2 || o == 3) dx = W - 1 - x;
      if (o == 3 || o == 4) dy = H - 1 - y;
      if (o == 5 || o == 8) dx = y;
      if (o == 6 || o == 7) dx = H - 1 - y;
      if (o == 5 || o == 6) dy = x;
      if (o == 7 || o == 8) dy = W - 1 - x;
      const std::size_t from = (static_cast<std::size_t>(y) * W + x) * c;
      const std::size_t to = (static_cast<std::size_t>(dy) * ow + dx) * c;
      for (int k = 0; k < c; ++k) out[to + k] = src[from + k];
    }
  }
  w = ow;
  h = oh;
  return out;
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
  const int o = exif_orientation(path);
  if (o != 1) img.pixels = orient(img.pixels.data(), img.width, img.height, 1, o);
  return img;
}

ImageRgb load_rgb(const std::filesystem::path& path) {
  ImageRgb img;
  const StbiPixels px = load(path, 3, img.width, img.height);
  const std::size_t n = static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
  const int o = exif_orientation(path);
  if (o != 1) {
    img.rgb = orient(px.get(), img.width, img.height, 3, o);
  } else {
    img.rgb.assign(px.get(), px.get() + 3 * n);
  }
  return img;
}

Image8 load_mask(const std::filesystem::path& path, bool object_white) {
  Image8 img;
  const StbiPixels px = load(path, 1, img.width, img.height);
  const std::size_t n = static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
  img.pixels.resize(n);
  if (object_white) {  // usual convention: object > 127, background dark
    for (std::size_t i = 0; i < n; ++i) img.pixels[i] = px.get()[i] > 127 ? 255 : 0;
  } else {
    for (std::size_t i = 0; i < n; ++i) img.pixels[i] = px.get()[i] == 0 ? 255 : 0;
  }
  return img;
}

}  // namespace sfm::data
