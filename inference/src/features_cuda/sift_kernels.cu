// CUDA SIFT kernels: thin wrappers around sift_core.hpp (compiled with -fmad=false, so the device
// evaluates exactly the operations the CPU reference does).
#include "sfm/features/sift_core.hpp"

#include "sift_kernels.cuh"

namespace sfm::features::cuda {

namespace {

namespace core = sift_core;

constexpr int kBlock = 256;

int blocks(long long n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

__global__ void k_to_float(const std::uint8_t* g, float* f, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f[i] = static_cast<float>(g[i]) / 255.0F;
}

__global__ void k_upsample2(const float* src, int sw, int sh, float* dst) {
  const int w = 2 * sw;
  const int X = blockIdx.x * blockDim.x + threadIdx.x;
  const int Y = blockIdx.y;
  if (X >= w || Y >= 2 * sh) return;
  dst[static_cast<long long>(Y) * w + X] = core::upsample_px(src, sw, sh, X, Y);
}

__global__ void k_downsample2(const float* src, int sw, float* dst, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (x >= w || y >= h) return;
  dst[static_cast<long long>(y) * w + x] = src[static_cast<long long>(2 * y) * sw + 2 * x];
}

__global__ void k_blur_h(const float* src, float* dst, int w, int h, const float* taps, int r) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (x >= w || y >= h) return;
  dst[static_cast<long long>(y) * w + x] = core::blur_h(src, w, x, y, taps, r);
}

__global__ void k_blur_v(const float* src, float* dst, int w, int h, const float* taps, int r) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y;
  if (x >= w || y >= h) return;
  dst[static_cast<long long>(y) * w + x] = core::blur_v(src, w, h, x, y, taps, r);
}

__global__ void k_difference(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = b[i] - a[i];
}

__global__ void k_detect(const float* dog, int w, int h, int S, int octave, float thr, float peak,
                         float edge, float sigma0, SiftCand* out, unsigned int* count,
                         unsigned int cap) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
  const int y = blockIdx.y + 1;
  const int d = blockIdx.z + 1;
  if (x > w - 2 || y > h - 2) return;
  const long long plane = static_cast<long long>(w) * h;
  const float* dm = dog + (d - 1) * plane;
  const float* d0 = dog + d * plane;
  const float* dp = dog + (d + 1) * plane;
  if (!core::is_extremum(dm, d0, dp, w, x, y, thr)) return;
  const core::Refined r = core::refine(dm, d0, dp, w, h, x, y, peak, edge);
  if (!r.ok) return;
  const float sn = static_cast<float>(d - 1) + r.b[2];
  if (sn < -1.0F || sn > static_cast<float>(S + 1)) return;
  const unsigned int k = atomicAdd(count, 1U);
  if (k >= cap) return;
  SiftCand c;
  c.octave = octave;
  c.dog = d;
  c.x0 = x;
  c.y0 = y;
  c.xi = r.x;
  c.yi = r.y;
  c.x = static_cast<float>(r.x) + r.b[0];
  c.y = static_cast<float>(r.y) + r.b[1];
  c.sigma = core::keypoint_sigma(sigma0, sn, S);
  c.response = fabsf(r.value);
  out[k] = c;
}

__global__ void k_describe(const float* levels, const long long* level_offset, const int* size,
                           int first_octave, int S, const SiftCand* cand, int n,
                           int max_orientations, int* count, float* angle, unsigned int* desc) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const SiftCand c = cand[k];
  const int oi = c.octave - first_octave;
  const int w = size[2 * oi];
  const int h = size[2 * oi + 1];
  const float* L = levels + level_offset[oi * (S + 3) + c.dog];
  float ang[4];
  const int no = core::orientations(L, w, h, c.x, c.y, c.sigma, ang, max_orientations);
  count[k] = no;
  float raw[core::kDescLen];
  for (int o = 0; o < no; ++o) {
    angle[4 * k + o] = ang[o];
    core::descriptor_raw(L, w, h, c.x, c.y, c.sigma, ang[o], raw);
    core::descriptor_finish(raw, desc + static_cast<long long>(4 * k + o) * 32);
  }
}

}  // namespace

void to_float(const std::uint8_t* d_gray, float* d_out, int n, cudaStream_t s) {
  k_to_float<<<blocks(n), kBlock, 0, s>>>(d_gray, d_out, n);
}

void upsample2(const float* d_src, int src_w, int src_h, float* d_dst, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(blocks(2LL * src_w)), static_cast<unsigned>(2 * src_h));
  k_upsample2<<<grid, kBlock, 0, s>>>(d_src, src_w, src_h, d_dst);
}

void downsample2(const float* d_src, int src_w, float* d_dst, int w, int h, cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(blocks(w)), static_cast<unsigned>(h));
  k_downsample2<<<grid, kBlock, 0, s>>>(d_src, src_w, d_dst, w, h);
}

void blur(const float* d_src, float* d_tmp, float* d_dst, int w, int h, const float* d_taps, int r,
          cudaStream_t s) {
  const dim3 grid(static_cast<unsigned>(blocks(w)), static_cast<unsigned>(h));
  k_blur_h<<<grid, kBlock, 0, s>>>(d_src, d_tmp, w, h, d_taps, r);
  k_blur_v<<<grid, kBlock, 0, s>>>(d_tmp, d_dst, w, h, d_taps, r);
}

void difference(const float* d_a, const float* d_b, float* d_out, int n, cudaStream_t s) {
  k_difference<<<blocks(n), kBlock, 0, s>>>(d_a, d_b, d_out, n);
}

void detect(const float* d_dog, int w, int h, int S, int octave, float threshold, float peak,
            float edge, float sigma0, SiftCand* d_out, unsigned int* d_count, unsigned int cap,
            cudaStream_t s) {
  if (w < 3 || h < 3) return;
  const dim3 grid(static_cast<unsigned>(blocks(w - 2)), static_cast<unsigned>(h - 2),
                  static_cast<unsigned>(S));
  k_detect<<<grid, kBlock, 0, s>>>(d_dog, w, h, S, octave, threshold, peak, edge, sigma0, d_out,
                                   d_count, cap);
}

void describe(const float* d_levels, const long long* d_level_offset, const int* d_size,
              int first_octave, int S, const SiftCand* d_cand, int n, int max_orientations,
              int* d_count, float* d_angle, unsigned int* d_desc, cudaStream_t s) {
  if (n == 0) return;
  // Per-thread descriptor histograms live in local memory: small blocks keep occupancy sane.
  k_describe<<<(n + 63) / 64, 64, 0, s>>>(d_levels, d_level_offset, d_size, first_octave, S, d_cand,
                                          n, max_orientations, d_count, d_angle, d_desc);
}

}  // namespace sfm::features::cuda
