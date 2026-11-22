// Orchestration of the CUDA ORB: device buffers, per-level kernel sequence, host-side selection.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cub/device/device_radix_sort.cuh>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"
#include "sfm/features/exact.hpp"
#include "sfm/features/orb_cuda.hpp"
#include "sfm/features/pyramid.hpp"

#include "orb_kernels.cuh"

namespace sfm::features {

namespace {

struct LevelTables {
  int w = 0;
  int h = 0;
  core::DeviceBuffer<int> x0, x1, wx, xn, y0, y1, wy, yn;
};

}  // namespace

struct OrbCuda::Impl {
  OrbParams p;
  int width = 0;
  int height = 0;
  std::vector<int> quotas;
  std::vector<LevelTables> tables;  // tables[l]: level l-1 -> level l (l >= 1); sizes for all l
  core::Stream stream;
  std::vector<core::DeviceBuffer<std::uint8_t>> d_level;
  std::vector<core::DeviceBuffer<std::uint8_t>> d_mask;
  core::DeviceBuffer<std::uint8_t> d_blur;
  core::DeviceBuffer<int> d_tmp;
  core::DeviceBuffer<std::uint8_t> d_score;
  core::DeviceBuffer<unsigned long long> d_keys;
  core::DeviceBuffer<unsigned long long> d_sorted;
  core::DeviceBuffer<unsigned int> d_count;
  core::DeviceBuffer<long long> d_harris;
  core::DeviceBuffer<unsigned int> d_xy;
  core::DeviceBuffer<long long> d_m;
  core::DeviceBuffer<unsigned int> d_desc;
  core::DeviceBuffer<signed char> d_pattern;
  core::DeviceBuffer<unsigned char> d_sort_temp;
  unsigned int cap = 0;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  float last_ms = 0.0F;

  ~Impl() {
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_stop != nullptr) cudaEventDestroy(ev_stop);
  }
};

OrbCuda::OrbCuda(const OrbParams& params, int width, int height) : impl_(std::make_unique<Impl>()) {
  if (params.edge_border < 22 || params.levels < 1 || !(params.scale > 1.0) || width < 1 ||
      height < 1 || width >= (1 << 20) || height >= (1 << 20)) {
    throw std::invalid_argument("OrbCuda: invalid parameters or image size");
  }
  Impl& m = *impl_;
  m.p = params;
  m.width = width;
  m.height = height;
  m.quotas = level_quotas(params);
  const auto L = static_cast<std::size_t>(params.levels);
  m.tables.resize(L);
  m.d_level.resize(L);
  m.d_mask.resize(L);
  for (std::size_t l = 0; l < L; ++l) {
    LevelTables& t = m.tables[l];
    const double f = std::pow(params.scale, static_cast<double>(l));
    t.w = l == 0 ? width : std::max(1, static_cast<int>(std::lround(width / f)));
    t.h = l == 0 ? height : std::max(1, static_cast<int>(std::lround(height / f)));
    const auto pixels = static_cast<std::size_t>(t.w) * static_cast<std::size_t>(t.h);
    m.d_level[l] = core::DeviceBuffer<std::uint8_t>(pixels);
    m.d_mask[l] = core::DeviceBuffer<std::uint8_t>(pixels);
    if (l == 0) continue;
    const LevelTables& prev = m.tables[l - 1];
    const AxisTable ax = axis_table(prev.w, t.w);
    const AxisTable ay = axis_table(prev.h, t.h);
    const auto up = [&](core::DeviceBuffer<int>& d, const std::vector<int>& h) {
      d = core::DeviceBuffer<int>(h.size());
      d.upload(h.data(), h.size(), m.stream);
    };
    up(t.x0, ax.i0);
    up(t.x1, ax.i1);
    up(t.wx, ax.w);
    up(t.xn, ax.nearest);
    up(t.y0, ay.i0);
    up(t.y1, ay.i1);
    up(t.wy, ay.w);
    up(t.yn, ay.nearest);
    m.stream.synchronize();  // the host tables go out of scope
  }
  const auto pixels0 = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  m.d_blur = core::DeviceBuffer<std::uint8_t>(pixels0);
  m.d_tmp = core::DeviceBuffer<int>(pixels0);
  m.d_score = core::DeviceBuffer<std::uint8_t>(pixels0);
  // Strict 3x3 maxima are never adjacent: at most ceil(w/2) ceil(h/2) per level.
  m.cap = static_cast<unsigned int>(((width + 1) / 2) * ((height + 1) / 2));
  m.d_keys = core::DeviceBuffer<unsigned long long>(m.cap);
  m.d_sorted = core::DeviceBuffer<unsigned long long>(m.cap);
  m.d_count = core::DeviceBuffer<unsigned int>(1);
  const int max_quota = *std::max_element(m.quotas.begin(), m.quotas.end());
  m.d_harris = core::DeviceBuffer<long long>(static_cast<std::size_t>(2 * std::max(1, max_quota)));
  m.d_xy = core::DeviceBuffer<unsigned int>(static_cast<std::size_t>(std::max(1, max_quota)));
  m.d_m = core::DeviceBuffer<long long>(static_cast<std::size_t>(2 * std::max(1, max_quota)));
  m.d_desc = core::DeviceBuffer<unsigned int>(static_cast<std::size_t>(8 * std::max(1, max_quota)));

  std::vector<signed char> pattern;
  for (const auto& t : rbrief_pattern()) pattern.insert(pattern.end(), t.begin(), t.end());
  m.d_pattern = core::DeviceBuffer<signed char>(pattern.size());
  m.d_pattern.upload(pattern.data(), pattern.size(), m.stream);

  std::size_t temp = 0;
  CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp, m.d_keys.data(), m.d_sorted.data(),
                                       static_cast<int>(m.cap), 0, 48, m.stream.get()));
  m.d_sort_temp = core::DeviceBuffer<unsigned char>(std::max<std::size_t>(temp, 1));
  CHECK(cudaEventCreate(&m.ev_start));
  CHECK(cudaEventCreate(&m.ev_stop));
  m.stream.synchronize();
}

OrbCuda::~OrbCuda() = default;
OrbCuda::OrbCuda(OrbCuda&&) noexcept = default;
OrbCuda& OrbCuda::operator=(OrbCuda&&) noexcept = default;

const OrbParams& OrbCuda::params() const { return impl_->p; }
int OrbCuda::width() const { return impl_->width; }
int OrbCuda::height() const { return impl_->height; }
float OrbCuda::last_device_ms() const { return impl_->last_ms; }

Features OrbCuda::detect(const data::Image8& gray, const data::Image8* mask) {
  Impl& m = *impl_;
  if (gray.width != m.width || gray.height != m.height ||
      (mask != nullptr && (mask->width != m.width || mask->height != m.height))) {
    throw std::invalid_argument("OrbCuda::detect: image size differs from the constructor's");
  }
  const cudaStream_t s = m.stream.get();
  CHECK(cudaEventRecord(m.ev_start, s));
  m.d_level[0].upload(gray.pixels.data(), gray.pixels.size(), m.stream);
  if (mask != nullptr) m.d_mask[0].upload(mask->pixels.data(), mask->pixels.size(), m.stream);
  for (std::size_t l = 1; l < m.tables.size(); ++l) {
    const LevelTables& t = m.tables[l];
    const int src_w = m.tables[l - 1].w;
    cuda::resize_bilinear(m.d_level[l - 1].data(), src_w, m.d_level[l].data(), t.w, t.h,
                          t.x0.data(), t.x1.data(), t.wx.data(), t.y0.data(), t.y1.data(),
                          t.wy.data(), s);
    if (mask != nullptr) {
      cuda::resize_nearest(m.d_mask[l - 1].data(), src_w, m.d_mask[l].data(), t.w, t.h, t.xn.data(),
                           t.yn.data(), s);
    }
  }
  CHECK(cudaGetLastError());

  Features out;
  const int border = m.p.edge_border;
  for (std::size_t l = 0; l < m.tables.size(); ++l) {
    const int w = m.tables[l].w;
    const int h = m.tables[l].h;
    const int quota = m.quotas[l];
    if (quota <= 0 || w <= 2 * border || h <= 2 * border) continue;
    const std::uint8_t* d_img = m.d_level[l].data();

    cuda::fast_scores(d_img, mask != nullptr ? m.d_mask[l].data() : nullptr, m.d_score.data(), w, h,
                      m.p.fast_threshold, border, s);
    CHECK(cudaMemsetAsync(m.d_count.data(), 0, sizeof(unsigned int), s));
    cuda::nms_compact(m.d_score.data(), w, h, m.d_keys.data(), m.d_count.data(), m.cap, s);
    CHECK(cudaGetLastError());
    unsigned int count = 0;
    m.d_count.download(&count, 1, m.stream);
    m.stream.synchronize();
    if (count > m.cap) throw std::runtime_error("OrbCuda: candidate buffer overflow");
    if (count == 0) continue;

    // First selection: FAST score descending, raster order on ties (= the CPU's stable_sort).
    std::size_t temp = m.d_sort_temp.size();
    CHECK(cub::DeviceRadixSort::SortKeys(m.d_sort_temp.data(), temp, m.d_keys.data(),
                                         m.d_sorted.data(), static_cast<int>(count), 0, 48, s));
    const int n = std::min(static_cast<int>(count), 2 * quota);
    cuda::harris(d_img, w, m.d_sorted.data(), n, m.d_harris.data(), s);
    cuda::blur7(d_img, m.d_tmp.data(), m.d_blur.data(), w, h, s);
    CHECK(cudaGetLastError());
    std::vector<unsigned long long> keys(static_cast<std::size_t>(n));
    std::vector<long long> hv(static_cast<std::size_t>(n));
    m.d_sorted.download(keys.data(), keys.size(), m.stream);
    m.d_harris.download(hv.data(), hv.size(), m.stream);
    m.stream.synchronize();

    // Second selection on the host (<= 2 x quota items): Harris descending, ties by (y, x).
    struct Cand {
      int x;
      int y;
      long long h;
    };
    std::vector<Cand> cand;
    cand.reserve(keys.size());
    for (std::size_t k = 0; k < keys.size(); ++k) {
      cand.push_back({static_cast<int>(keys[k] & 0xFFFFFULL),
                      static_cast<int>((keys[k] >> 20) & 0xFFFFFULL), hv[k]});
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
      if (a.h != b.h) return a.h > b.h;
      return a.y != b.y ? a.y < b.y : a.x < b.x;
    });
    cand.resize(std::min(cand.size(), static_cast<std::size_t>(quota)));
    std::vector<unsigned int> xy;
    xy.reserve(cand.size());
    for (const Cand& c : cand)
      xy.push_back((static_cast<unsigned int>(c.y) << 20) | static_cast<unsigned int>(c.x));
    const int k = static_cast<int>(xy.size());
    m.d_xy.upload(xy.data(), xy.size(), m.stream);
    cuda::describe(d_img, m.d_blur.data(), w, m.d_xy.data(), k, m.d_pattern.data(), m.d_m.data(),
                   m.d_desc.data(), s);
    CHECK(cudaGetLastError());
    std::vector<long long> mom(2 * static_cast<std::size_t>(k));
    std::vector<unsigned int> desc(8 * static_cast<std::size_t>(k));
    m.d_m.download(mom.data(), mom.size(), m.stream);
    m.d_desc.download(desc.data(), desc.size(), m.stream);
    m.stream.synchronize();

    const double sx = static_cast<double>(m.width) / w;
    const double sy = static_cast<double>(m.height) / h;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      out.xy.push_back(static_cast<float>((cand[i].x + 0.5) * sx - 0.5));
      out.xy.push_back(static_cast<float>((cand[i].y + 0.5) * sy - 0.5));
      out.angle.push_back(angle_deg(mom[2 * i], mom[2 * i + 1]));
      out.response.push_back(exact::harris_response(cand[i].h));
      out.octave.push_back(static_cast<std::uint8_t>(l));
      out.desc.insert(out.desc.end(), desc.begin() + static_cast<std::ptrdiff_t>(8 * i),
                      desc.begin() + static_cast<std::ptrdiff_t>(8 * i + 8));
    }
  }
  CHECK(cudaEventRecord(m.ev_stop, s));
  CHECK(cudaEventSynchronize(m.ev_stop));
  CHECK(cudaEventElapsedTime(&m.last_ms, m.ev_start, m.ev_stop));
  return out;
}

}  // namespace sfm::features
