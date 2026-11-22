// Orchestration of the CUDA SIFT: device buffers, per-octave kernel sequence, host-side ordering,
// mask and selection (the CPU's functions), per-keypoint kernel, assembly.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"
#include "sfm/features/sift_cuda.hpp"

#include "sift_kernels.cuh"

namespace sfm::features {

namespace {

constexpr unsigned int kMaxCandidates = 1U << 20;

struct Taps {
  long long offset = -1;  // into d_taps; -1 = no blur
  int r = 0;
};

}  // namespace

struct SiftCuda::Impl {
  SiftParams p;
  int width = 0;
  int height = 0;
  int S = 0;
  float sigma0 = 0.0F;
  std::vector<int> ow;  // octave sizes
  std::vector<int> oh;
  std::vector<long long> level_offset;  // [oi * (S + 3) + l] into d_levels
  std::vector<long long> dog_offset;    // [oi] into d_dog (S + 2 planes)
  std::vector<std::vector<Taps>> taps;  // [oi][l]
  core::Stream stream;
  core::DeviceBuffer<std::uint8_t> d_gray;
  core::DeviceBuffer<float> d_base;
  core::DeviceBuffer<float> d_start;
  core::DeviceBuffer<float> d_tmp;
  core::DeviceBuffer<float> d_levels;
  core::DeviceBuffer<float> d_dog;
  core::DeviceBuffer<float> d_taps;
  core::DeviceBuffer<long long> d_level_offset;
  core::DeviceBuffer<int> d_size;
  core::DeviceBuffer<cuda::SiftCand> d_cand;
  core::DeviceBuffer<unsigned int> d_count;
  core::DeviceBuffer<cuda::SiftCand> d_sel;
  core::DeviceBuffer<int> d_ocount;
  core::DeviceBuffer<float> d_angle;
  core::DeviceBuffer<unsigned int> d_desc;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  float last_ms = 0.0F;

  ~Impl() {
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
  }
};

SiftCuda::SiftCuda(const SiftParams& params, int width, int height)
    : impl_(std::make_unique<Impl>()) {
  check_sift_params(params);
  if (width < 1 || height < 1 || width >= (1 << 15) || height >= (1 << 15)) {
    throw std::invalid_argument("SiftCuda: invalid image size");
  }
  Impl& m = *impl_;
  m.p = params;
  m.width = width;
  m.height = height;
  m.S = params.octave_resolution;
  m.sigma0 = sift_detail::sigma0(m.S);
  const int levels = m.S + 3;
  long long lv = 0;
  long long dg = 0;
  long long biggest = 0;
  std::vector<float> taps;
  for (int oi = 0; oi < params.octaves; ++oi) {
    int w = 0;
    int h = 0;
    if (!sift_detail::octave_size(width, height, params.first_octave + oi, w, h)) break;
    m.ow.push_back(w);
    m.oh.push_back(h);
    const long long plane = static_cast<long long>(w) * h;
    biggest = std::max(biggest, plane);
    for (int l = 0; l < levels; ++l) m.level_offset.push_back(lv + l * plane);
    lv += levels * plane;
    m.dog_offset.push_back(dg);
    dg += (levels - 1) * plane;
    std::vector<Taps> t;
    for (const auto& k : octave_blur_taps(params, params.first_octave + oi)) {
      Taps e;
      if (!k.empty()) {
        e.offset = static_cast<long long>(taps.size());
        e.r = static_cast<int>(k.size() / 2);
        taps.insert(taps.end(), k.begin(), k.end());
      }
      t.push_back(e);
    }
    m.taps.push_back(std::move(t));
  }
  const auto plane0 = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  m.d_gray = core::DeviceBuffer<std::uint8_t>(plane0);
  m.d_base = core::DeviceBuffer<float>(plane0);
  if (m.ow.empty()) return;  // too small: detect returns no keypoints
  m.d_start = core::DeviceBuffer<float>(static_cast<std::size_t>(biggest));
  m.d_tmp = core::DeviceBuffer<float>(static_cast<std::size_t>(biggest));
  m.d_levels = core::DeviceBuffer<float>(static_cast<std::size_t>(lv));
  m.d_dog = core::DeviceBuffer<float>(static_cast<std::size_t>(dg));
  m.d_taps = core::DeviceBuffer<float>(taps.size());
  m.d_taps.upload(taps.data(), taps.size(), m.stream);
  m.d_level_offset = core::DeviceBuffer<long long>(m.level_offset.size());
  m.d_level_offset.upload(m.level_offset.data(), m.level_offset.size(), m.stream);
  std::vector<int> size;
  for (std::size_t oi = 0; oi < m.ow.size(); ++oi) {
    size.push_back(m.ow[oi]);
    size.push_back(m.oh[oi]);
  }
  m.d_size = core::DeviceBuffer<int>(size.size());
  m.d_size.upload(size.data(), size.size(), m.stream);
  m.d_cand = core::DeviceBuffer<cuda::SiftCand>(kMaxCandidates);
  m.d_count = core::DeviceBuffer<unsigned int>(1);
  const std::size_t sel = params.max_features;
  m.d_sel = core::DeviceBuffer<cuda::SiftCand>(sel);
  m.d_ocount = core::DeviceBuffer<int>(sel);
  m.d_angle = core::DeviceBuffer<float>(4 * sel);
  m.d_desc = core::DeviceBuffer<unsigned int>(4 * sel * kSiftWords);
  CHECK(cudaEventCreate(&m.ev_start));
  CHECK(cudaEventCreate(&m.ev_stop));
  m.stream.synchronize();
}

SiftCuda::~SiftCuda() = default;
SiftCuda::SiftCuda(SiftCuda&&) noexcept = default;
SiftCuda& SiftCuda::operator=(SiftCuda&&) noexcept = default;
const SiftParams& SiftCuda::params() const { return impl_->p; }
int SiftCuda::width() const { return impl_->width; }
int SiftCuda::height() const { return impl_->height; }
float SiftCuda::last_device_ms() const { return impl_->last_ms; }

Features SiftCuda::detect(const data::Image8& gray, const data::Image8* mask) {
  Impl& m = *impl_;
  if (gray.width != m.width || gray.height != m.height ||
      (mask != nullptr && (mask->width != m.width || mask->height != m.height))) {
    throw std::invalid_argument("SiftCuda::detect: image size differs from the constructor's");
  }
  Features out;
  out.words = kSiftWords;
  if (m.ow.empty()) return out;
  const cudaStream_t s = m.stream.get();
  const int S = m.S;
  const int levels = S + 3;
  CHECK(cudaEventRecord(m.ev_start, s));
  m.d_gray.upload(gray.pixels.data(), gray.pixels.size(), m.stream);
  cuda::to_float(m.d_gray.data(), m.d_base.data(), static_cast<int>(gray.pixels.size()), s);
  CHECK(cudaMemsetAsync(m.d_count.data(), 0, sizeof(unsigned int), s));
  float* L = m.d_levels.data();
  for (std::size_t oi = 0; oi < m.ow.size(); ++oi) {
    const int w = m.ow[oi];
    const int h = m.oh[oi];
    const int o = m.p.first_octave + static_cast<int>(oi);
    const auto lvl = [&](int l) {
      return L + m.level_offset[oi * levels + static_cast<std::size_t>(l)];
    };
    const Taps& t0 = m.taps[oi][0];
    float* start = t0.offset < 0 ? lvl(0) : m.d_start.data();
    if (oi == 0 && o == -1) {
      cuda::upsample2(m.d_base.data(), m.width, m.height, start, s);
    } else if (oi == 0) {
      CHECK(cudaMemcpyAsync(start, m.d_base.data(), m.d_base.bytes(), cudaMemcpyDeviceToDevice, s));
    } else {
      const float* prev = L + m.level_offset[(oi - 1) * levels + static_cast<std::size_t>(S)];
      cuda::downsample2(prev, m.ow[oi - 1], start, w, h, s);
    }
    if (t0.offset >= 0) {
      cuda::blur(start, m.d_tmp.data(), lvl(0), w, h, m.d_taps.data() + t0.offset, t0.r, s);
    }
    for (int l = 1; l < levels; ++l) {
      const Taps& t = m.taps[oi][static_cast<std::size_t>(l)];
      cuda::blur(lvl(l - 1), m.d_tmp.data(), lvl(l), w, h, m.d_taps.data() + t.offset, t.r, s);
    }
    const long long plane = static_cast<long long>(w) * h;
    float* dog = m.d_dog.data() + m.dog_offset[oi];
    for (int d = 0; d < levels - 1; ++d) {
      cuda::difference(lvl(d), lvl(d + 1), dog + d * plane, static_cast<int>(plane), s);
    }
    cuda::detect(dog, w, h, S, o, 0.8F * m.p.peak_threshold, m.p.peak_threshold, m.p.edge_threshold,
                 m.sigma0, m.d_cand.data(), m.d_count.data(), kMaxCandidates, s);
    CHECK(cudaGetLastError());
  }
  unsigned int count = 0;
  m.d_count.download(&count, 1, m.stream);
  m.stream.synchronize();
  if (count > kMaxCandidates) {
    throw std::runtime_error("SiftCuda: " + std::to_string(count) +
                             " extrema exceed the buffer of " + std::to_string(kMaxCandidates));
  }
  std::vector<cuda::SiftCand> raw(count);
  if (count > 0) m.d_cand.download(raw.data(), count, m.stream);
  m.stream.synchronize();

  // The CPU's detection order (octave, level, raster of the detection pixel) numbers the
  // candidates; then the CPU's mask test and selection.
  std::sort(raw.begin(), raw.end(), [](const cuda::SiftCand& a, const cuda::SiftCand& b) {
    return std::tie(a.octave, a.dog, a.y0, a.x0) < std::tie(b.octave, b.dog, b.y0, b.x0);
  });
  std::vector<sift_detail::Candidate> cand;
  for (std::uint32_t k = 0; k < raw.size(); ++k) {
    const cuda::SiftCand& r = raw[k];
    sift_detail::Candidate c;
    c.octave = r.octave;
    c.dog = r.dog;
    c.xi = r.xi;
    c.yi = r.yi;
    c.x = r.x;
    c.y = r.y;
    c.sigma = r.sigma;
    c.response = r.response;
    c.seq = k;
    if (sift_detail::mask_allows(mask, c)) cand.push_back(c);
  }
  sift_detail::select_candidates(cand, m.p.max_features);
  const int n = static_cast<int>(cand.size());
  if (n == 0) return out;
  std::vector<cuda::SiftCand> sel(cand.size());
  for (std::size_t k = 0; k < cand.size(); ++k) {
    const auto& c = cand[k];
    sel[k] = {c.octave, c.dog, 0, 0, c.xi, c.yi, c.x, c.y, c.sigma, c.response};
  }
  m.d_sel.upload(sel.data(), sel.size(), m.stream);
  // Slots past a location's orientation count are not written by the kernel: zero them so the
  // whole prefix that is downloaded is defined.
  CHECK(cudaMemsetAsync(m.d_angle.data(), 0, 4 * sel.size() * sizeof(float), s));
  CHECK(cudaMemsetAsync(m.d_desc.data(), 0, 4 * sel.size() * kSiftWords * sizeof(unsigned int), s));
  cuda::describe(m.d_levels.data(), m.d_level_offset.data(), m.d_size.data(), m.p.first_octave, S,
                 m.d_sel.data(), n, std::min(m.p.max_orientations, 4), m.d_ocount.data(),
                 m.d_angle.data(), m.d_desc.data(), s);
  CHECK(cudaGetLastError());
  std::vector<int> ocount(cand.size());
  std::vector<float> angle(4 * cand.size());
  std::vector<unsigned int> desc(4 * cand.size() * kSiftWords);
  m.d_ocount.download(ocount.data(), ocount.size(), m.stream);
  m.d_angle.download(angle.data(), angle.size(), m.stream);
  m.d_desc.download(desc.data(), desc.size(), m.stream);
  CHECK(cudaEventRecord(m.ev_stop, s));
  m.stream.synchronize();
  CHECK(cudaEventElapsedTime(&m.last_ms, m.ev_start, m.ev_stop));
  for (std::size_t k = 0; k < cand.size(); ++k) {
    for (int o = 0; o < ocount[k]; ++o) {
      sift_detail::append_keypoint(out, cand[k], angle[4 * k + static_cast<std::size_t>(o)],
                                   desc.data() + (4 * k + static_cast<std::size_t>(o)) * kSiftWords,
                                   m.p.first_octave);
    }
  }
  return out;
}

}  // namespace sfm::features
