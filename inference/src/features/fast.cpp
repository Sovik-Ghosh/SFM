#include "sfm/features/fast.hpp"

#include <algorithm>
#include <stdexcept>

#include "sfm/features/exact.hpp"

namespace sfm::features {

namespace {

std::size_t index(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
         static_cast<std::size_t>(x);
}

}  // namespace

int fast_score(const data::Image8& img, int x, int y) {
  return exact::fast_score(img.pixels.data(), img.width, x, y);
}

std::vector<Corner> nms_3x3(const std::vector<int>& scores, int width, int height) {
  if (scores.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
    throw std::invalid_argument("nms_3x3: score map size != width * height");
  }
  std::vector<Corner> out;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int s = scores[index(x, y, width)];
      if (s <= 0) continue;
      bool keep = true;
      for (int dy = -1; dy <= 1 && keep; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = x + dx;
          const int ny = y + dy;
          if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const int n = scores[index(nx, ny, width)];
          const bool before = dy < 0 || (dy == 0 && dx < 0);  // earlier in raster order
          if (n > s || (before && n == s)) {
            keep = false;
            break;
          }
        }
      }
      if (keep) out.push_back({x, y, s});
    }
  }
  return out;
}

std::vector<Corner> detect_fast(const data::Image8& img, const data::Image8* mask, int threshold,
                                int border) {
  if (border < 3) throw std::invalid_argument("detect_fast: border must be >= 3");
  if (mask != nullptr && (mask->width != img.width || mask->height != img.height)) {
    throw std::invalid_argument("detect_fast: mask size differs from the image");
  }
  std::vector<int> scores(img.pixels.size(), 0);
  for (int y = border; y < img.height - border; ++y) {
    for (int x = border; x < img.width - border; ++x) {
      if (mask != nullptr && mask->pixels[index(x, y, img.width)] == 0) continue;
      const int s = fast_score(img, x, y);
      if (s > threshold) scores[index(x, y, img.width)] = s;
    }
  }
  return nms_3x3(scores, img.width, img.height);
}

std::int64_t harris_int(const data::Image8& img, int x, int y) {
  return exact::harris_int(img.pixels.data(), img.width, x, y);
}

float harris_response(const data::Image8& img, int x, int y) {
  return exact::harris_response(harris_int(img, x, y));
}

}  // namespace sfm::features
