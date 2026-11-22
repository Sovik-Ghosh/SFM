#pragma once
// Per-pixel and per-keypoint arithmetic of SIFT (sift.hpp), shared by the CPU reference and the
// CUDA kernels so both produce identical bits. Rules: only correctly rounded float operations
// written out in a fixed order (the CUDA files are compiled with -fmad=false, the host compiler
// does not contract), transcendental functions from exact_math, loops in a fixed order.
// Images are row-major float, width w, height h.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "sfm/features/exact_math.hpp"

namespace sfm::features::sift_core {

inline constexpr int kOriBins = 36;
inline constexpr int kNbp = 4;  // spatial bins per side
inline constexpr int kNbo = 8;  // orientation bins
inline constexpr int kDescLen = kNbp * kNbp * kNbo;
inline constexpr float kTwoPi = 2.0F * exact_math::kPi;

SFM_HD inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
SFM_HD inline int mini(int a, int b) { return a < b ? a : b; }
SFM_HD inline int maxi(int a, int b) { return a > b ? a : b; }

// Pixel (X, Y) of the image upsampled x2 (size 2w x 2h): even coordinates copy the source,
// odd ones average the two neighbours (the last row / column repeats), rows first.
SFM_HD inline float upsample_px(const float* src, int w, int h, int X, int Y) {
  const int x = X >> 1;
  const int y = Y >> 1;
  const int x1 = mini(x + 1, w - 1);
  const int y1 = mini(y + 1, h - 1);
  const float* r0 = src + static_cast<std::ptrdiff_t>(y) * w;
  const float* r1 = src + static_cast<std::ptrdiff_t>(y1) * w;
  const bool hx = (X & 1) != 0;
  const bool hy = (Y & 1) != 0;
  const float a = hx ? 0.5F * (r0[x] + r0[x1]) : r0[x];
  if (!hy) return a;
  const float b = hx ? 0.5F * (r1[x] + r1[x1]) : r1[x];
  return 0.5F * (a + b);
}

// Horizontal / vertical pass of a separable blur with taps[0 .. 2 r], replicated borders.
SFM_HD inline float blur_h(const float* src, int w, int x, int y, const float* taps, int r) {
  const float* row = src + static_cast<std::ptrdiff_t>(y) * w;
  float acc = 0.0F;
  for (int k = -r; k <= r; ++k) acc = acc + taps[k + r] * row[clampi(x + k, 0, w - 1)];
  return acc;
}
SFM_HD inline float blur_v(const float* src, int w, int h, int x, int y, const float* taps, int r) {
  float acc = 0.0F;
  for (int k = -r; k <= r; ++k) {
    acc = acc + taps[k + r] * src[static_cast<std::ptrdiff_t>(clampi(y + k, 0, h - 1)) * w + x];
  }
  return acc;
}

// Strict 3x3x3 extremum of DoG level d0 at (x, y), 1 <= x <= w - 2, 1 <= y <= h - 2:
// v >= thr and above all 26 neighbours, or v <= -thr and below all of them.
SFM_HD inline bool is_extremum(const float* dm, const float* d0, const float* dp, int w, int x,
                               int y, float thr) {
  const std::ptrdiff_t c = static_cast<std::ptrdiff_t>(y) * w + x;
  const float v = d0[c];
  const bool is_max = v >= thr;
  const bool is_min = v <= -thr;
  if (!is_max && !is_min) return false;
  const float* lv[3] = {dm, d0, dp};
  for (int l = 0; l < 3; ++l) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (l == 1 && dx == 0 && dy == 0) continue;
        const float n = lv[l][c + static_cast<std::ptrdiff_t>(dy) * w + dx];
        if (is_max ? !(v > n) : !(v < n)) return false;
      }
    }
  }
  return true;
}

// Solves A b = r (3x3, row-major A) by Gaussian elimination with partial pivoting; false (b = 0)
// if a pivot is below 1e-10.
SFM_HD inline bool solve3(const float A_in[9], const float r[3], float b[3]) {
  float A[3][4];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) A[i][j] = A_in[3 * i + j];
    A[i][3] = r[i];
  }
  for (int j = 0; j < 3; ++j) {
    int pi = j;
    float pv = A[j][j];
    for (int i = j + 1; i < 3; ++i) {
      if (fabsf(A[i][j]) > fabsf(pv)) {
        pv = A[i][j];
        pi = i;
      }
    }
    if (fabsf(pv) < 1e-10F) {
      b[0] = b[1] = b[2] = 0.0F;
      return false;
    }
    for (int k = j; k < 4; ++k) {
      const float t = A[j][k];
      A[j][k] = A[pi][k];
      A[pi][k] = t;
    }
    for (int k = j; k < 4; ++k) A[j][k] = A[j][k] / pv;
    for (int i = j + 1; i < 3; ++i) {
      const float f = A[i][j];
      for (int k = j; k < 4; ++k) A[i][k] = A[i][k] - f * A[j][k];
    }
  }
  for (int i = 2; i >= 0; --i) {
    float v = A[i][3];
    for (int k = i + 1; k < 3; ++k) v = v - A[i][k] * b[k];
    b[i] = v;
  }
  return true;
}

struct Refined {
  int x = 0;  // integer position after the steps
  int y = 0;
  float b[3] = {0.0F, 0.0F, 0.0F};  // offset (x, y, s)
  float value = 0.0F;               // |D(x^)| is the response
  bool ok = false;
};

SFM_HD inline float at(const float* d, int w, int x, int y) {
  return d[static_cast<std::ptrdiff_t>(y) * w + x];
}

// VLFeat's refinement of an extremum at (x, y) of DoG level d0 (dm / dp: the levels below /
// above): <= 5 Newton steps on the 3D quadratic, moving one pixel when |offset| > 0.6; then the
// peak, edge and offset tests.
SFM_HD inline Refined refine(const float* dm, const float* d0, const float* dp, int w, int h, int x,
                             int y, float peak_threshold, float edge_threshold) {
  Refined out;
  int dx = 0;
  int dy = 0;
  float g[3] = {0.0F, 0.0F, 0.0F};
  float H[9] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  float b[3] = {0.0F, 0.0F, 0.0F};
  float c = 0.0F;
  for (int iter = 0; iter < 5; ++iter) {
    x += dx;
    y += dy;
    c = at(d0, w, x, y);
    g[0] = 0.5F * (at(d0, w, x + 1, y) - at(d0, w, x - 1, y));
    g[1] = 0.5F * (at(d0, w, x, y + 1) - at(d0, w, x, y - 1));
    g[2] = 0.5F * (at(dp, w, x, y) - at(dm, w, x, y));
    H[0] = at(d0, w, x + 1, y) + at(d0, w, x - 1, y) - 2.0F * c;
    H[4] = at(d0, w, x, y + 1) + at(d0, w, x, y - 1) - 2.0F * c;
    H[8] = at(dp, w, x, y) + at(dm, w, x, y) - 2.0F * c;
    H[1] = H[3] = 0.25F * (at(d0, w, x + 1, y + 1) + at(d0, w, x - 1, y - 1) -
                           at(d0, w, x - 1, y + 1) - at(d0, w, x + 1, y - 1));
    H[2] = H[6] = 0.25F * (at(dp, w, x + 1, y) + at(dm, w, x - 1, y) - at(dm, w, x + 1, y) -
                           at(dp, w, x - 1, y));
    H[5] = H[7] = 0.25F * (at(dp, w, x, y + 1) + at(dm, w, x, y - 1) - at(dm, w, x, y + 1) -
                           at(dp, w, x, y - 1));
    const float r[3] = {-g[0], -g[1], -g[2]};
    solve3(H, r, b);
    dx = ((b[0] > 0.6F && x < w - 2) ? 1 : 0) + ((b[0] < -0.6F && x > 1) ? -1 : 0);
    dy = ((b[1] > 0.6F && y < h - 2) ? 1 : 0) + ((b[1] < -0.6F && y > 1) ? -1 : 0);
    if (dx == 0 && dy == 0) break;
  }
  const float value = c + 0.5F * (g[0] * b[0] + g[1] * b[1] + g[2] * b[2]);
  const float tr = H[0] + H[4];
  const float score = (tr * tr) / (H[0] * H[4] - H[1] * H[1]);
  const float edge = ((edge_threshold + 1.0F) * (edge_threshold + 1.0F)) / edge_threshold;
  const float xn = static_cast<float>(x) + b[0];
  const float yn = static_cast<float>(y) + b[1];
  out.x = x;
  out.y = y;
  out.b[0] = b[0];
  out.b[1] = b[1];
  out.b[2] = b[2];
  out.value = value;
  out.ok = fabsf(value) > peak_threshold && score < edge && score >= 0.0F && fabsf(b[0]) < 1.5F &&
           fabsf(b[1]) < 1.5F && fabsf(b[2]) < 1.5F && xn >= 0.0F &&
           xn <= static_cast<float>(w - 1) && yn >= 0.0F && yn <= static_cast<float>(h - 1);
  return out;
}

// Keypoint scale in octave pixels at refined level sn: sigma0 2^(sn / S).
SFM_HD inline float keypoint_sigma(float sigma0, float sn, int S) {
  return sigma0 * exact_math::exp(sn / static_cast<float>(S) * 0.693147180559945F);
}

// Gradient magnitude and angle [0, 2 pi) at an interior pixel (central differences).
SFM_HD inline void gradient(const float* L, int w, int x, int y, float& mod, float& ang) {
  const float gx = 0.5F * (at(L, w, x + 1, y) - at(L, w, x - 1, y));
  const float gy = 0.5F * (at(L, w, x, y + 1) - at(L, w, x, y - 1));
  mod = sqrtf(gx * gx + gy * gy);
  ang = exact_math::atan2_2pi(gy, gx);
}

// Up to max_out dominant orientations (radians, [0, 2 pi)) at (x, y) of Gaussian level L with
// keypoint scale sigma (octave pixels). Returns the count.
SFM_HD inline int orientations(const float* L, int w, int h, float x, float y, float sigma,
                               float* out, int max_out) {
  const float sigmaw = 1.5F * sigma;
  const int W = maxi(static_cast<int>(floorf(3.0F * sigmaw)), 1);
  const int xi = static_cast<int>(floorf(x + 0.5F));
  const int yi = static_cast<int>(floorf(y + 0.5F));
  if (xi < 0 || xi > w - 1 || yi < 0 || yi > h - 1) return 0;
  float hist[kOriBins];
  for (float& v : hist) v = 0.0F;
  const float lim = static_cast<float>(W * W) + 0.6F;
  const float den = 2.0F * sigmaw * sigmaw;
  for (int dyi = maxi(-W, 1 - yi); dyi <= mini(W, h - 2 - yi); ++dyi) {
    for (int dxi = maxi(-W, 1 - xi); dxi <= mini(W, w - 2 - xi); ++dxi) {
      const float dx = static_cast<float>(xi + dxi) - x;
      const float dy = static_cast<float>(yi + dyi) - y;
      const float r2 = dx * dx + dy * dy;
      if (r2 >= lim) continue;
      const float wgt = exact_math::exp(-r2 / den);
      float mod = 0.0F;
      float ang = 0.0F;
      gradient(L, w, xi + dxi, yi + dyi, mod, ang);
      const float fbin = static_cast<float>(kOriBins) * ang / kTwoPi;
      const int bin = static_cast<int>(floorf(fbin - 0.5F));
      const float rbin = fbin - static_cast<float>(bin) - 0.5F;
      const float mw = mod * wgt;
      hist[(bin + kOriBins) % kOriBins] = hist[(bin + kOriBins) % kOriBins] + (1.0F - rbin) * mw;
      hist[(bin + 1) % kOriBins] = hist[(bin + 1) % kOriBins] + rbin * mw;
    }
  }
  for (int it = 0; it < 6; ++it) {  // circular [1 1 1] / 3, in place as VLFeat
    float prev = hist[kOriBins - 1];
    const float first = hist[0];
    for (int i = 0; i < kOriBins - 1; ++i) {
      const float nh = (prev + hist[i] + hist[i + 1]) / 3.0F;
      prev = hist[i];
      hist[i] = nh;
    }
    hist[kOriBins - 1] = (prev + hist[kOriBins - 1] + first) / 3.0F;
  }
  float maxh = 0.0F;
  for (const float v : hist) maxh = v > maxh ? v : maxh;
  int n = 0;
  for (int i = 0; i < kOriBins && n < max_out; ++i) {
    const float h0 = hist[i];
    const float hm = hist[(i + kOriBins - 1) % kOriBins];
    const float hp = hist[(i + 1) % kOriBins];
    if (h0 > 0.8F * maxh && h0 > hm && h0 > hp) {
      const float di = -0.5F * (hp - hm) / (hp + hm - 2.0F * h0);
      float th = kTwoPi * (static_cast<float>(i) + di + 0.5F) / static_cast<float>(kOriBins);
      if (th >= kTwoPi) th = th - kTwoPi;
      if (th < 0.0F) th = th + kTwoPi;
      out[n++] = th;
    }
  }
  return n;
}

// Raw 4x4x8 histogram (before normalisation) at (x, y), scale sigma (octave pixels), angle a.
SFM_HD inline void descriptor_raw(const float* L, int w, int h, float x, float y, float sigma,
                                  float a, float* d) {
  for (int k = 0; k < kDescLen; ++k) d[k] = 0.0F;
  const float sbp = 3.0F * sigma;  // magnif 3
  const int W = static_cast<int>(floorf(1.41421356F * sbp * 2.5F + 0.5F));
  const int xi = static_cast<int>(floorf(x + 0.5F));
  const int yi = static_cast<int>(floorf(y + 0.5F));
  float st = 0.0F;
  float ct = 0.0F;
  exact_math::sincos(a, st, ct);
  const float half = 0.5F * static_cast<float>(kNbp);  // = 2 exactly
  const float wden = 2.0F * half * half;               // window sigma = NBP / 2 bins
  for (int dyi = maxi(-W, 1 - yi); dyi <= mini(W, h - 2 - yi); ++dyi) {
    for (int dxi = maxi(-W, 1 - xi); dxi <= mini(W, w - 2 - xi); ++dxi) {
      float mod = 0.0F;
      float ang = 0.0F;
      gradient(L, w, xi + dxi, yi + dyi, mod, ang);
      float theta = ang - a;
      if (theta < 0.0F) theta = theta + kTwoPi;
      if (theta >= kTwoPi) theta = theta - kTwoPi;
      const float dx = static_cast<float>(xi + dxi) - x;
      const float dy = static_cast<float>(yi + dyi) - y;
      const float nx = (ct * dx + st * dy) / sbp;
      const float ny = (-st * dx + ct * dy) / sbp;
      const float nt = static_cast<float>(kNbo) * theta / kTwoPi;
      const float win = exact_math::exp(-(nx * nx + ny * ny) / wden);
      const int binx = static_cast<int>(floorf(nx - 0.5F));
      const int biny = static_cast<int>(floorf(ny - 0.5F));
      const int bint = static_cast<int>(floorf(nt));
      const float rbinx = nx - (static_cast<float>(binx) + 0.5F);
      const float rbiny = ny - (static_cast<float>(biny) + 0.5F);
      const float rbint = nt - static_cast<float>(bint);
      const float base = win * mod;
      for (int dbx = 0; dbx < 2; ++dbx) {
        const int bx = binx + dbx;
        if (bx < -kNbp / 2 || bx >= kNbp / 2) continue;
        const float wx = fabsf(1.0F - static_cast<float>(dbx) - rbinx);
        for (int dby = 0; dby < 2; ++dby) {
          const int by = biny + dby;
          if (by < -kNbp / 2 || by >= kNbp / 2) continue;
          const float wy = fabsf(1.0F - static_cast<float>(dby) - rbiny);
          for (int dbt = 0; dbt < 2; ++dbt) {
            const float wt = fabsf(1.0F - static_cast<float>(dbt) - rbint);
            const int k = ((by + kNbp / 2) * kNbp + (bx + kNbp / 2)) * kNbo + (bint + dbt) % kNbo;
            d[k] = d[k] + base * wx * wy * wt;
          }
        }
      }
    }
  }
}

// Normalise, clamp 0.2, renormalise, RootSIFT, quantise (512 v rounded, <= 255) and pack 4 bytes
// per word (byte k -> word k / 4, bits 8 (k % 4)).
SFM_HD inline void descriptor_finish(float* d, std::uint32_t* out) {
  float n2 = 0.0F;
  for (int k = 0; k < kDescLen; ++k) n2 = n2 + d[k] * d[k];
  float n = sqrtf(n2);
  if (n > 0.0F) {
    for (int k = 0; k < kDescLen; ++k) d[k] = d[k] / n;
  }
  for (int k = 0; k < kDescLen; ++k) d[k] = d[k] < 0.2F ? d[k] : 0.2F;
  n2 = 0.0F;
  for (int k = 0; k < kDescLen; ++k) n2 = n2 + d[k] * d[k];
  n = sqrtf(n2);
  if (n > 0.0F) {
    for (int k = 0; k < kDescLen; ++k) d[k] = d[k] / n;
  }
  float l1 = 0.0F;
  for (int k = 0; k < kDescLen; ++k) l1 = l1 + d[k];
  for (int k = 0; k < kDescLen / 4; ++k) out[k] = 0U;
  for (int k = 0; k < kDescLen; ++k) {
    const float v = l1 > 0.0F ? sqrtf(d[k] / l1) : 0.0F;
    float q = floorf(512.0F * v + 0.5F);
    q = q < 255.0F ? q : 255.0F;
    out[k / 4] |= static_cast<std::uint32_t>(q) << (8 * (k % 4));
  }
}

}  // namespace sfm::features::sift_core
