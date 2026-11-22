#pragma once
// Point colouring shared by the CPU scene builder and the CUDA-OpenGL interop kernel (viewer V5):
// the same code, built only from correctly rounded IEEE float operations (on the device through
// __fadd_rn & co., which nvcc never fuses into FMAs; the host has no FMA on x86-64 without
// -march), so both give the same bytes.

#include <cstdint>

#include "sfm/features/exact_math.hpp"  // SFM_HD, mul / add / sub / div

namespace sfm::viewer::hd {

namespace em = sfm::features::exact_math;

// The value a point is coloured by (ColorMode order: gray, height, error, track length).
SFM_HD inline float point_value(int mode, float x, float y, float z, float ux, float uy, float uz,
                                std::uint8_t gray, float error_px, int track_length) {
  switch (mode) {
    case 0:
      return em::div(static_cast<float>(gray), 255.0F);
    case 1:  // height along the world up: (x ux + y uy) + z uz
      return em::add(em::add(em::mul(x, ux), em::mul(y, uy)), em::mul(z, uz));
    case 2:
      return error_px;
    default:
      return static_cast<float>(track_length);
  }
}

// (v - lo) / (hi - lo) clamped to [0, 1]; NaN -> 0.
SFM_HD inline float normalise(float v, float lo, float hi) {
  const float t = em::div(em::sub(v, lo), em::sub(hi, lo));
  if (!(t > 0.0F)) return 0.0F;  // also NaN
  return t < 1.0F ? t : 1.0F;
}

// Viridis (matplotlib, 5 stops at 0, .25, .5, .75, 1, linear between, rounded half away from
// zero) of t in [0, 1].
SFM_HD inline void viridis(float t, std::uint8_t out[3]) {
  constexpr float kStops[5][3] = {
      {68, 1, 84}, {59, 82, 139}, {33, 145, 140}, {94, 201, 98}, {253, 231, 37}};
  float v = t;
  if (!(v > 0.0F)) v = 0.0F;  // NaN -> 0
  if (v > 1.0F) v = 1.0F;
  const float x = em::mul(v, 4.0F);
  const int i = x < 3.999F ? static_cast<int>(x) : 3;
  const float f = em::sub(x, static_cast<float>(i));
  for (int c = 0; c < 3; ++c) {
    const float a = kStops[i][c];
    const float b = kStops[i + 1][c];
    const float y = em::add(a, em::mul(f, em::sub(b, a)));  // in [0, 255]
    // lround for y >= 0, exactly: integer part, then the (exact) fraction against one half.
    const int k = static_cast<int>(y);
    const float frac = em::sub(y, static_cast<float>(k));
    out[c] = static_cast<std::uint8_t>(frac >= 0.5F ? k + 1 : k);
  }
}

}  // namespace sfm::viewer::hd
