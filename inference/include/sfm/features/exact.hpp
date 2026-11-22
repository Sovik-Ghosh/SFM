#pragma once
// Per-pixel / per-keypoint arithmetic of the ORB detector, shared by the CPU code and the CUDA
// kernels so both produce bit-identical results (P3 gate: exact parity). Everything is integer
// except the steering of the descriptor, which uses correctly rounded double operations only
// (sqrt, division, and multiply / subtract without FMA contraction: __dmul_rn / __dsub_rn on the
// device, plain operators on the host, where x86-64 and ISO-mode GCC do not contract).

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define SFM_HD __host__ __device__
#else
#define SFM_HD
#endif

namespace sfm::features::exact {

// Bresenham circle of radius 3, clockwise from the top (FAST).
SFM_HD inline void circle_offset(int i, int& dx, int& dy) {
  constexpr int kDx[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
  constexpr int kDy[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
  dx = kDx[i];
  dy = kDy[i];
}

// FAST score: max over the 16 arcs of 9 contiguous circle pixels of min |p - c| on the arc
// (bright and dark arcs). The pixel is a corner for threshold t exactly when the score > t.
SFM_HD inline int fast_score(const std::uint8_t* img, int stride, int x, int y) {
  const int c = img[static_cast<std::ptrdiff_t>(y) * stride + x];
  int d[16];
  for (int i = 0; i < 16; ++i) {
    int dx = 0;
    int dy = 0;
    circle_offset(i, dx, dy);
    d[i] = img[static_cast<std::ptrdiff_t>(y + dy) * stride + x + dx] - c;
  }
  int best = 0;
  for (int s = 0; s < 16; ++s) {
    int bright = 255;
    int dark = 255;
    for (int k = 0; k < 9; ++k) {
      const int v = d[(s + k) & 15];
      bright = v < bright ? v : bright;
      dark = -v < dark ? -v : dark;
    }
    best = bright > best ? bright : best;
    best = dark > best ? dark : best;
  }
  return best;
}

// Integer Harris response over the 7x7 window of 3x3 Sobel gradients centred on (x, y):
// 25 (a b - c^2) - (a + b)^2 = 25 (det M - 0.04 trace(M)^2), exact in int64
// (|a|, |b|, |c| <= 49 * 1020^2 < 2^26).
SFM_HD inline std::int64_t harris_int(const std::uint8_t* img, int stride, int x, int y) {
  std::int64_t a = 0;
  std::int64_t b = 0;
  std::int64_t c = 0;
  for (int v = y - 3; v <= y + 3; ++v) {
    for (int u = x - 3; u <= x + 3; ++u) {
      const std::uint8_t* p = img + static_cast<std::ptrdiff_t>(v) * stride + u;
      const int ix = (p[1 - stride] + 2 * p[1] + p[1 + stride]) -
                     (p[-1 - stride] + 2 * p[-1] + p[-1 + stride]);
      const int iy = (p[stride - 1] + 2 * p[stride] + p[stride + 1]) -
                     (p[-stride - 1] + 2 * p[-stride] + p[-stride + 1]);
      a += static_cast<std::int64_t>(ix) * ix;
      b += static_cast<std::int64_t>(iy) * iy;
      c += static_cast<std::int64_t>(ix) * iy;
    }
  }
  return 25 * (a * b - c * c) - (a + b) * (a + b);
}

// harris_int scaled like OpenCV's ORB response (gradients / (4 * 7 * 255), k = 0.04): one
// correctly rounded multiplication, so host and device agree.
SFM_HD inline float harris_response(std::int64_t h) {
  constexpr double kScale = 1.0 / (4.0 * 7.0 * 255.0);
  constexpr double kS4 = kScale * kScale * kScale * kScale / 25.0;
  return static_cast<float>(static_cast<double>(h) * kS4);
}

// Intensity moments m10 = sum u I, m01 = sum v I over the radius-15 disc around (x, y).
SFM_HD inline void moments(const std::uint8_t* img, int stride, int x, int y, std::int64_t& m10,
                           std::int64_t& m01) {
  // Half-widths of the disc rows: floor(sqrt(225 - v^2 + 0.5)) for v = 0..15.
  constexpr int kUmax[16] = {15, 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 9, 7, 5, 0};
  m10 = 0;
  m01 = 0;
  for (int v = -15; v <= 15; ++v) {
    const int umax = kUmax[v < 0 ? -v : v];
    const std::uint8_t* row = img + static_cast<std::ptrdiff_t>(y + v) * stride + x;
    for (int u = -umax; u <= umax; ++u) {
      const int i = row[u];
      m10 += static_cast<std::int64_t>(u) * i;
      m01 += static_cast<std::int64_t>(v) * i;
    }
  }
}

// Unit direction (cos, sin) of the moment vector; (1, 0) when both moments are 0.
SFM_HD inline void steering(std::int64_t m10, std::int64_t m01, double& c, double& s) {
  const auto a = static_cast<double>(m10);  // |m| < 2^21: exact
  const auto b = static_cast<double>(m01);
#ifdef __CUDA_ARCH__
  const double r = __dsqrt_rn(__dadd_rn(__dmul_rn(a, a), __dmul_rn(b, b)));
  c = r > 0.0 ? __ddiv_rn(a, r) : 1.0;
  s = r > 0.0 ? __ddiv_rn(b, r) : 0.0;
#else
  const double r = std::sqrt(a * a + b * b);
  c = r > 0.0 ? a / r : 1.0;
  s = r > 0.0 ? b / r : 0.0;
#endif
}

// Pattern offset (px, py) rotated by (c, s) and rounded half away from zero.
SFM_HD inline void rotate_round(int px, int py, double c, double s, int& rx, int& ry) {
#ifdef __CUDA_ARCH__
  rx = static_cast<int>(llround(__dsub_rn(__dmul_rn(px, c), __dmul_rn(py, s))));
  ry = static_cast<int>(llround(__dadd_rn(__dmul_rn(px, s), __dmul_rn(py, c))));
#else
  rx = static_cast<int>(std::llround(px * c - py * s));
  ry = static_cast<int>(std::llround(px * s + py * c));
#endif
}

// Bilinear interpolation with 11-bit weights (w = weight of the second sample, 0..2048):
// ((p00 (2048 - wx) + p01 wx) (2048 - wy) + (p10 (2048 - wx) + p11 wx) wy + 2^21) >> 22.
SFM_HD inline std::uint8_t bilinear(int p00, int p01, int p10, int p11, int wx, int wy) {
  const int top = p00 * (2048 - wx) + p01 * wx;
  const int bottom = p10 * (2048 - wx) + p11 * wx;
  return static_cast<std::uint8_t>((top * (2048 - wy) + bottom * wy + (1 << 21)) >> 22);
}

// 7-tap integer Gaussian (sigma 2): round(256 exp(-k^2 / 8) / sum), centre adjusted to sum 256.
SFM_HD inline int blur_weight(int k) {  // k = -3..3
  constexpr int kW[7] = {18, 34, 49, 54, 49, 34, 18};
  return kW[k + 3];
}

}  // namespace sfm::features::exact
