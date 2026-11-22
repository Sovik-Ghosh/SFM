#pragma once
// Small synthetic images for the feature tests.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "sfm/data/image_io.hpp"

namespace sfm::features::test {

inline data::Image8 make_image(int w, int h, const std::function<int(int, int)>& f) {
  data::Image8 img;
  img.width = w;
  img.height = h;
  img.pixels.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int v = f(x, y);
      img.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
  }
  return img;
}

inline int at(const data::Image8& img, int x, int y) {
  return img.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) +
                    static_cast<std::size_t>(x)];
}

// Smooth random texture: a sum of fixed sinusoids (deterministic, no RNG).
inline int texture(int x, int y) {
  const double v = 128 + 40 * std::sin(0.31 * x + 0.17 * y) + 30 * std::sin(0.07 * x - 0.23 * y) +
                   25 * std::cos(0.19 * x * 0.5 + 0.41 * y) +
                   20 * std::sin(0.53 * x + 0.05 * y * y / 40);
  return static_cast<int>(std::lround(v));
}

// Image rotated by +90 degrees about the centre of a square image: out(x, y) = in(y, n - 1 - x).
// A point (u, v) of `in` lands on (n - 1 - v, u) — in image coordinates (y down) this is a
// rotation that adds +90 degrees to atan2(dy, dx).
inline data::Image8 rotate90(const data::Image8& in) {
  const int n = in.width;
  return make_image(n, n, [&](int x, int y) { return at(in, y, n - 1 - x); });
}

}  // namespace sfm::features::test
