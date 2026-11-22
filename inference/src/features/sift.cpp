// SIFT CPU reference (sift.hpp). The arithmetic is in sift_core.hpp, shared with the CUDA version.
#include "sfm/features/sift.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>

#include "sfm/features/sift_core.hpp"

namespace sfm::features {

namespace core = sift_core;
using sift_detail::Candidate;

std::vector<float> gaussian_taps(double sigma) {
  const int r = std::max(1, static_cast<int>(std::ceil(4.0 * sigma)));
  std::vector<double> t(static_cast<std::size_t>(2 * r + 1));
  double sum = 0.0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    const double k = static_cast<double>(i) - r;
    t[i] = std::exp(-0.5 * k * k / (sigma * sigma));
    sum += t[i];
  }
  std::vector<float> out(t.size());
  for (std::size_t k = 0; k < t.size(); ++k) out[k] = static_cast<float>(t[k] / sum);
  return out;
}

std::vector<std::vector<float>> octave_blur_taps(const SiftParams& p, int octave) {
  const int S = p.octave_resolution;
  const int s_min = -1;
  const double sigmak = std::pow(2.0, 1.0 / S);
  const double sigma0 = 1.6 * sigmak;
  const double dsigma0 = sigma0 * std::sqrt(1.0 - 1.0 / (sigmak * sigmak));
  std::vector<std::vector<float>> taps(static_cast<std::size_t>(S + 3));
  if (octave == p.first_octave) {
    const double sa = sigma0 * std::pow(sigmak, s_min);
    const double sb = 0.5 * std::pow(2.0, -p.first_octave);  // nominal 0.5 px image blur
    if (sa > sb) taps[0] = gaussian_taps(std::sqrt(sa * sa - sb * sb));
  }  // later octaves start from level S - 1 (= sigma0 2^(s_min / S) after halving): no blur
  for (int s = s_min + 1; s <= S + 1; ++s) {
    taps[static_cast<std::size_t>(s - s_min)] = gaussian_taps(dsigma0 * std::pow(sigmak, s));
  }
  return taps;
}

namespace sift_detail {

float sigma0(int octave_resolution) {
  return static_cast<float>(1.6 * std::pow(2.0, 1.0 / octave_resolution));
}

bool octave_size(int width, int height, int octave, int& w, int& h) {
  w = octave < 0 ? width << -octave : width >> octave;
  h = octave < 0 ? height << -octave : height >> octave;
  return w >= 8 && h >= 8;
}

void select_candidates(std::vector<Candidate>& c, std::uint32_t max_features) {
  std::sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
    if (a.response != b.response) return a.response > b.response;
    return std::tie(a.octave, a.dog, a.yi, a.xi, a.seq) <
           std::tie(b.octave, b.dog, b.yi, b.xi, b.seq);
  });
  if (c.size() > max_features) c.resize(max_features);
}

bool mask_allows(const data::Image8* mask, const Candidate& c) {
  if (mask == nullptr) return true;
  const int W = mask->width;
  const int H = mask->height;
  const int mx =
      std::clamp(static_cast<int>(std::floor(std::ldexp(c.x, c.octave) + 0.5F)), 0, W - 1);
  const int my =
      std::clamp(static_cast<int>(std::floor(std::ldexp(c.y, c.octave) + 0.5F)), 0, H - 1);
  return mask->pixels[static_cast<std::size_t>(my) * static_cast<std::size_t>(W) +
                      static_cast<std::size_t>(mx)] != 0;
}

void append_keypoint(Features& out, const Candidate& c, float angle_rad, const std::uint32_t* desc,
                     int first_octave) {
  out.xy.push_back(std::ldexp(c.x, c.octave));
  out.xy.push_back(std::ldexp(c.y, c.octave));
  float deg = angle_rad * (180.0F / exact_math::kPi);
  if (deg >= 360.0F) deg = 0.0F;
  out.angle.push_back(deg);
  out.response.push_back(c.response);
  out.octave.push_back(static_cast<std::uint8_t>(c.octave - first_octave));
  out.desc.insert(out.desc.end(), desc, desc + kSiftWords);
}

}  // namespace sift_detail

namespace {

using Plane = std::vector<float>;

Plane blur(const Plane& src, int w, int h, const std::vector<float>& taps) {
  const int r = static_cast<int>(taps.size() / 2);
  Plane tmp(src.size());
  Plane dst(src.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      tmp[static_cast<std::size_t>(y) * w + x] = core::blur_h(src.data(), w, x, y, taps.data(), r);
    }
  }
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      dst[static_cast<std::size_t>(y) * w + x] =
          core::blur_v(tmp.data(), w, h, x, y, taps.data(), r);
    }
  }
  return dst;
}

struct Octave {
  int o = 0;
  int w = 0;
  int h = 0;
  std::vector<Plane> L;  // S + 3 Gaussian levels
};

}  // namespace

void check_sift_params(const SiftParams& p) {
  if (p.first_octave < -1 || p.first_octave > 0) {
    throw std::invalid_argument("SIFT: first_octave must be -1 or 0");
  }
  if (p.octave_resolution < 1 || p.octaves < 1 || p.max_orientations < 1 ||
      p.max_orientations > 4) {
    throw std::invalid_argument(
        "SIFT: octaves, octave_resolution >= 1 and max_orientations in [1, 4] required");
  }
}

Features detect_sift(const data::Image8& gray, const data::Image8* mask, const SiftParams& p) {
  check_sift_params(p);
  Features out;
  out.words = kSiftWords;
  const int W = gray.width;
  const int H = gray.height;
  const int S = p.octave_resolution;
  const float sigma0 = sift_detail::sigma0(S);
  Plane base(static_cast<std::size_t>(W) * static_cast<std::size_t>(H));
  for (std::size_t k = 0; k < base.size(); ++k) {
    base[k] = static_cast<float>(gray.pixels[k]) / 255.0F;
  }

  std::vector<Octave> octaves;
  std::vector<Candidate> cand;
  std::uint32_t seq = 0;
  for (int oi = 0; oi < p.octaves; ++oi) {
    Octave oc;
    oc.o = p.first_octave + oi;
    if (!sift_detail::octave_size(W, H, oc.o, oc.w, oc.h)) break;
    const int w = oc.w;
    const int h = oc.h;
    Plane start(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (oi == 0 && oc.o == -1) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          start[static_cast<std::size_t>(y) * w + x] = core::upsample_px(base.data(), W, H, x, y);
        }
      }
    } else if (oi == 0) {
      start = base;
    } else {
      const Octave& pr = octaves.back();
      const Plane& src = pr.L[static_cast<std::size_t>(S)];  // level s = S - 1
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          start[static_cast<std::size_t>(y) * w + x] =
              src[2 * static_cast<std::size_t>(y) * static_cast<std::size_t>(pr.w) +
                  2 * static_cast<std::size_t>(x)];
        }
      }
    }
    const auto taps = octave_blur_taps(p, oc.o);
    oc.L.push_back(taps[0].empty() ? std::move(start) : blur(start, w, h, taps[0]));
    for (int l = 1; l < S + 3; ++l) {
      oc.L.push_back(blur(oc.L.back(), w, h, taps[static_cast<std::size_t>(l)]));
    }
    std::vector<Plane> D;
    for (int d = 0; d < S + 2; ++d) {
      Plane v(oc.L[0].size());
      for (std::size_t k = 0; k < v.size(); ++k) {
        v[k] = oc.L[static_cast<std::size_t>(d) + 1][k] - oc.L[static_cast<std::size_t>(d)][k];
      }
      D.push_back(std::move(v));
    }
    const float thr = 0.8F * p.peak_threshold;
    for (int d = 1; d <= S; ++d) {
      const float* dm = D[static_cast<std::size_t>(d - 1)].data();
      const float* d0 = D[static_cast<std::size_t>(d)].data();
      const float* dp = D[static_cast<std::size_t>(d) + 1].data();
      for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
          if (!core::is_extremum(dm, d0, dp, w, x, y, thr)) continue;
          const core::Refined r =
              core::refine(dm, d0, dp, w, h, x, y, p.peak_threshold, p.edge_threshold);
          if (!r.ok) continue;
          const float sn = static_cast<float>(d - 1) + r.b[2];  // s = d + s_min
          if (sn < -1.0F || sn > static_cast<float>(S + 1)) continue;
          Candidate c;
          c.octave = oc.o;
          c.dog = d;
          c.xi = r.x;
          c.yi = r.y;
          c.x = static_cast<float>(r.x) + r.b[0];
          c.y = static_cast<float>(r.y) + r.b[1];
          c.sigma = core::keypoint_sigma(sigma0, sn, S);
          c.response = std::fabs(r.value);
          c.seq = seq++;  // counted before the mask test (the CUDA version numbers the same way)
          if (sift_detail::mask_allows(mask, c)) cand.push_back(c);
        }
      }
    }
    octaves.push_back(std::move(oc));
  }

  sift_detail::select_candidates(cand, p.max_features);
  float raw[core::kDescLen];
  float ang[4];
  std::uint32_t q[kSiftWords];
  for (const Candidate& c : cand) {
    const Octave& oc = octaves[static_cast<std::size_t>(c.octave - p.first_octave)];
    const float* L = oc.L[static_cast<std::size_t>(c.dog)].data();
    const int n =
        core::orientations(L, oc.w, oc.h, c.x, c.y, c.sigma, ang, std::min(p.max_orientations, 4));
    for (int k = 0; k < n; ++k) {
      core::descriptor_raw(L, oc.w, oc.h, c.x, c.y, c.sigma, ang[k], raw);
      core::descriptor_finish(raw, q);
      sift_detail::append_keypoint(out, c, ang[k], q, p.first_octave);
    }
  }
  return out;
}

}  // namespace sfm::features
