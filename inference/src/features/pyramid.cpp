#include "sfm/features/pyramid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "sfm/features/exact.hpp"

namespace sfm::features {

namespace {

std::size_t index(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
         static_cast<std::size_t>(x);
}

data::Image8 blank(int width, int height) {
  data::Image8 out;
  out.width = width;
  out.height = height;
  out.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
  return out;
}

}  // namespace

AxisTable axis_table(int src, int dst) {
  if (src < 1 || dst < 1) throw std::invalid_argument("axis_table: sizes must be >= 1");
  AxisTable t;
  t.i0.resize(static_cast<std::size_t>(dst));
  t.i1.resize(static_cast<std::size_t>(dst));
  t.w.resize(static_cast<std::size_t>(dst));
  t.nearest.resize(static_cast<std::size_t>(dst));
  const double scale = static_cast<double>(src) / dst;
  for (int x = 0; x < dst; ++x) {
    const auto k = static_cast<std::size_t>(x);
    const double f = std::clamp((x + 0.5) * scale - 0.5, 0.0, src - 1.0);
    t.i0[k] = static_cast<int>(f);
    t.i1[k] = std::min(t.i0[k] + 1, src - 1);
    t.w[k] = static_cast<int>(std::lround((f - t.i0[k]) * 2048.0));
    t.nearest[k] = std::min(static_cast<int>((x + 0.5) * scale), src - 1);
  }
  return t;
}

data::Image8 resize(const data::Image8& src, int width, int height, Resample method) {
  if (width < 1 || height < 1 || src.width < 1 || src.height < 1) {
    throw std::invalid_argument("resize: image sizes must be >= 1");
  }
  const AxisTable tx = axis_table(src.width, width);
  const AxisTable ty = axis_table(src.height, height);
  data::Image8 dst = blank(width, height);
  const auto p = [&](int x, int y) { return static_cast<int>(src.pixels[index(x, y, src.width)]); };
  for (int y = 0; y < height; ++y) {
    const auto ky = static_cast<std::size_t>(y);
    for (int x = 0; x < width; ++x) {
      const auto kx = static_cast<std::size_t>(x);
      if (method == Resample::kNearest) {
        dst.pixels[index(x, y, width)] =
            static_cast<std::uint8_t>(p(tx.nearest[kx], ty.nearest[ky]));
        continue;
      }
      dst.pixels[index(x, y, width)] =
          exact::bilinear(p(tx.i0[kx], ty.i0[ky]), p(tx.i1[kx], ty.i0[ky]), p(tx.i0[kx], ty.i1[ky]),
                          p(tx.i1[kx], ty.i1[ky]), tx.w[kx], ty.w[ky]);
    }
  }
  return dst;
}

std::vector<data::Image8> build_pyramid(const data::Image8& base, int levels, double scale,
                                        Resample method) {
  if (levels < 1 || !(scale > 1.0)) {
    throw std::invalid_argument("build_pyramid: levels >= 1 and scale > 1 required");
  }
  std::vector<data::Image8> out;
  out.reserve(static_cast<std::size_t>(levels));
  out.push_back(base);
  for (int l = 1; l < levels; ++l) {
    const double f = std::pow(scale, l);
    const int w = std::max(1, static_cast<int>(std::lround(base.width / f)));
    const int h = std::max(1, static_cast<int>(std::lround(base.height / f)));
    out.push_back(resize(out.back(), w, h, method));
  }
  return out;
}

data::Image8 gaussian_blur7(const data::Image8& src) {
  const int w = src.width;
  const int h = src.height;
  std::vector<int> tmp(src.pixels.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int acc = 0;
      for (int k = -3; k <= 3; ++k) {
        acc += exact::blur_weight(k) * src.pixels[index(std::clamp(x + k, 0, w - 1), y, w)];
      }
      tmp[index(x, y, w)] = acc;
    }
  }
  data::Image8 dst = blank(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int acc = 0;
      for (int k = -3; k <= 3; ++k) {
        acc += exact::blur_weight(k) * tmp[index(x, std::clamp(y + k, 0, h - 1), w)];
      }
      dst.pixels[index(x, y, w)] = static_cast<std::uint8_t>((acc + (1 << 15)) >> 16);
    }
  }
  return dst;
}

}  // namespace sfm::features
