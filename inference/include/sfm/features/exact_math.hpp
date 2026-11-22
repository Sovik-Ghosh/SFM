#pragma once
// Float exp / atan2 / sin / cos that give the same bits on the host and on the device (P7 SIFT
// gate: CPU and CUDA descriptors identical). Built only from correctly rounded IEEE operations:
// +, -, *, / (on the device through __fadd_rn & co., which nvcc never fuses into FMAs; on the host
// x86-64 without -march=... has no FMA and C++ without GNU extensions does not contract), floor
// and exact scaling by powers of two. The polynomials are Cephes' single-precision ones
// (S. Moshier); errors stay within a few ulp of the true values (tests in test_exact_math.cpp).

#include <cmath>

#ifdef __CUDACC__
#define SFM_HD __host__ __device__
#else
#define SFM_HD
#endif

namespace sfm::features::exact_math {

SFM_HD inline float mul(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
SFM_HD inline float add(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
SFM_HD inline float sub(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}
SFM_HD inline float div(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}

inline constexpr float kPi = 3.14159265358979323846F;

// e^x for x <= 88; 0 below -87.
SFM_HD inline float exp(float x) {
  if (x < -87.0F) return 0.0F;
  if (x > 88.0F) x = 88.0F;
  const float n = floorf(add(mul(x, 1.44269504088896341F), 0.5F));
  // Cody-Waite: ln 2 = 0.693359375 - 2.12194440e-4 (n * 0.693359375 is exact for |n| < 2^15).
  const float r = sub(sub(x, mul(n, 0.693359375F)), mul(n, -2.12194440e-4F));
  float p = 1.9875691500e-4F;
  p = add(mul(p, r), 1.3981999507e-3F);
  p = add(mul(p, r), 8.3334519073e-3F);
  p = add(mul(p, r), 4.1665795894e-2F);
  p = add(mul(p, r), 1.6666665459e-1F);
  p = add(mul(p, r), 5.0000001201e-1F);
  const float y = add(add(mul(mul(p, r), r), r), 1.0F);
  return ldexpf(y, static_cast<int>(n));  // exact
}

// atan(x) for x >= 0.
SFM_HD inline float atan_pos(float x) {
  float y0 = 0.0F;
  if (x > 2.414213562373095F) {
    y0 = 0.5F * kPi;
    x = div(-1.0F, x);
  } else if (x > 0.4142135623730950F) {
    y0 = 0.25F * kPi;
    x = div(sub(x, 1.0F), add(x, 1.0F));
  }
  const float z = mul(x, x);
  float p = 8.05374449538e-2F;
  p = sub(mul(p, z), 1.38776856032e-1F);
  p = add(mul(p, z), 1.99777106478e-1F);
  p = sub(mul(p, z), 3.33329491539e-1F);
  return add(add(mul(mul(p, z), x), x), y0);
}

// atan2(y, x) in [0, 2 pi) (SIFT orientations); 0 for (0, 0).
SFM_HD inline float atan2_2pi(float y, float x) {
  const float ax = fabsf(x);
  const float ay = fabsf(y);
  if (ax == 0.0F && ay == 0.0F) return 0.0F;
  // Angle of (ax, ay) in [0, pi/2], from the smaller ratio for accuracy.
  const float a = ay <= ax ? atan_pos(div(ay, ax)) : sub(0.5F * kPi, atan_pos(div(ax, ay)));
  float t = a;
  if (x < 0.0F) t = sub(kPi, a);
  if (y < 0.0F) t = sub(2.0F * kPi, t);
  return t >= 2.0F * kPi ? 0.0F : t;
}

// sin and cos of x, |x| <= 8192 (Cephes sinf / cosf range reduction by pi/4).
SFM_HD inline void sincos(float x, float& s, float& c) {
  float sign_s = 1.0F;
  if (x < 0.0F) {
    x = -x;
    sign_s = -1.0F;
  }
  int j = static_cast<int>(floorf(mul(x, 1.27323954473516F)));  // 4 / pi
  float y = static_cast<float>(j);
  if ((j & 1) != 0) {
    j += 1;
    y = add(y, 1.0F);
  }
  j &= 7;
  const float r = sub(sub(sub(x, mul(y, 0.78515625F)), mul(y, 2.4187564849853515625e-4F)),
                      mul(y, 3.77489497744594108e-8F));
  const float z = mul(r, r);
  float ps = -1.9515295891e-4F;
  ps = add(mul(ps, z), 8.3321608736e-3F);
  ps = sub(mul(ps, z), 1.6666654611e-1F);
  const float sr = add(mul(mul(ps, z), r), r);
  float pc = 2.443315711809948e-5F;
  pc = sub(mul(pc, z), 1.388731625493765e-3F);
  pc = add(mul(pc, z), 4.166664568298827e-2F);
  const float cr = add(sub(mul(mul(pc, z), z), mul(0.5F, z)), 1.0F);
  // Octant j in {0, 2, 4, 6}: (sin, cos) of x = rotations of (sr, cr).
  float so = sr;
  float co = cr;
  if (j == 2) {
    so = cr;
    co = -sr;
  } else if (j == 4) {
    so = -sr;
    co = -cr;
  } else if (j == 6) {
    so = -cr;
    co = sr;
  }
  s = sign_s * so;
  c = co;
}

}  // namespace sfm::features::exact_math
