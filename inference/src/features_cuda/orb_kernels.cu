// CUDA kernels of the ORB detector. The arithmetic is exact.hpp's, so results equal the CPU's.
#include "sfm/core/cuda_check.h"
#include "sfm/features/exact.hpp"

#include "orb_kernels.cuh"

namespace sfm::features::cuda {

namespace {

constexpr int kBlock = 256;
constexpr unsigned int kFull = 0xffffffffU;

int blocks(long long n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

__global__ void resize_bilinear_kernel(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst,
                                       int dst_w, int dst_h, const int* d_x0, const int* d_x1,
                                       const int* d_wx, const int* d_y0, const int* d_y1,
                                       const int* d_wy) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(dst_w) * dst_h) return;
  const int x = static_cast<int>(i % dst_w);
  const int y = static_cast<int>(i / dst_w);
  const std::uint8_t* r0 = d_src + static_cast<long long>(d_y0[y]) * src_w;
  const std::uint8_t* r1 = d_src + static_cast<long long>(d_y1[y]) * src_w;
  d_dst[i] = exact::bilinear(r0[d_x0[x]], r0[d_x1[x]], r1[d_x0[x]], r1[d_x1[x]], d_wx[x], d_wy[y]);
}

__global__ void resize_nearest_kernel(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst,
                                      int dst_w, int dst_h, const int* d_xn, const int* d_yn) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(dst_w) * dst_h) return;
  const int x = static_cast<int>(i % dst_w);
  const int y = static_cast<int>(i / dst_w);
  d_dst[i] = d_src[static_cast<long long>(d_yn[y]) * src_w + d_xn[x]];
}

__global__ void blur_h_kernel(const std::uint8_t* d_src, int* d_tmp, int w, int h) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(w) * h) return;
  const int x = static_cast<int>(i % w);
  const long long row = i - x;
  int acc = 0;
  for (int k = -3; k <= 3; ++k)
    acc += exact::blur_weight(k) * d_src[row + min(max(x + k, 0), w - 1)];
  d_tmp[i] = acc;
}

__global__ void blur_v_kernel(const int* d_tmp, std::uint8_t* d_dst, int w, int h) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(w) * h) return;
  const int x = static_cast<int>(i % w);
  const int y = static_cast<int>(i / w);
  int acc = 0;
  for (int k = -3; k <= 3; ++k) {
    acc += exact::blur_weight(k) * d_tmp[static_cast<long long>(min(max(y + k, 0), h - 1)) * w + x];
  }
  d_dst[i] = static_cast<std::uint8_t>((acc + (1 << 15)) >> 16);
}

__global__ void fast_kernel(const std::uint8_t* d_img, const std::uint8_t* d_mask,
                            std::uint8_t* d_score, int w, int h, int threshold, int border) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(w) * h) return;
  const int x = static_cast<int>(i % w);
  const int y = static_cast<int>(i / w);
  std::uint8_t out = 0;
  if (x >= border && x < w - border && y >= border && y < h - border &&
      (d_mask == nullptr || d_mask[i] != 0)) {
    const int s = exact::fast_score(d_img, w, x, y);
    if (s > threshold) out = static_cast<std::uint8_t>(s);
  }
  d_score[i] = out;
}

__global__ void nms_kernel(const std::uint8_t* d_score, int w, int h, unsigned long long* d_keys,
                           unsigned int* d_count, unsigned int cap) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= static_cast<long long>(w) * h) return;
  const int s = d_score[i];
  if (s == 0) return;
  const int x = static_cast<int>(i % w);
  const int y = static_cast<int>(i / w);
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = x + dx;
      const int ny = y + dy;
      if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const int n = d_score[static_cast<long long>(ny) * w + nx];
      const bool before = dy < 0 || (dy == 0 && dx < 0);
      if (n > s || (before && n == s)) return;
    }
  }
  const unsigned int slot = atomicAdd(d_count, 1U);
  if (slot < cap) {
    d_keys[slot] = (static_cast<unsigned long long>(255 - s) << 40) |
                   (static_cast<unsigned long long>(y) << 20) | static_cast<unsigned long long>(x);
  }
}

__global__ void harris_kernel(const std::uint8_t* d_img, int w, const unsigned long long* d_keys,
                              int n, long long* d_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const unsigned long long k = d_keys[i];
  const int x = static_cast<int>(k & 0xFFFFFULL);
  const int y = static_cast<int>((k >> 20) & 0xFFFFFULL);
  d_out[i] = exact::harris_int(d_img, w, x, y);
}

// One warp per keypoint.
__global__ void describe_kernel(const std::uint8_t* d_img, const std::uint8_t* d_blur, int w,
                                const unsigned int* d_xy, int n, const signed char* d_pattern,
                                long long* d_m, unsigned int* d_desc) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp >= n) return;  // whole warps exit together (n is per warp)
  const unsigned int xy = d_xy[warp];
  const int x = static_cast<int>(xy & 0xFFFFFU);
  const int y = static_cast<int>(xy >> 20);

  // Moments: rows v = -15..15 spread over the lanes (lane handles v = lane - 15, lane 31 idle).
  constexpr int kUmax[16] = {15, 14, 14, 14, 14, 14, 13, 13, 12, 12, 11, 10, 9, 7, 5, 0};
  long long m10 = 0;
  long long m01 = 0;
  if (lane < 31) {
    const int v = lane - 15;
    const int umax = kUmax[v < 0 ? -v : v];
    const std::uint8_t* row = d_img + static_cast<long long>(y + v) * w + x;
    for (int u = -umax; u <= umax; ++u) {
      const int iv = row[u];
      m10 += static_cast<long long>(u) * iv;
      m01 += static_cast<long long>(v) * iv;
    }
  }
  for (int off = 16; off > 0; off /= 2) {
    m10 += __shfl_down_sync(kFull, m10, off);
    m01 += __shfl_down_sync(kFull, m01, off);
  }
  m10 = __shfl_sync(kFull, m10, 0);
  m01 = __shfl_sync(kFull, m01, 0);
  double c = 1.0;
  double s = 0.0;
  exact::steering(m10, m01, c, s);

  // Test i = 32 k + lane; word k = ballot over the lanes.
  for (int k = 0; k < 8; ++k) {
    const signed char* p = d_pattern + 4 * (32 * k + lane);
    int ax = 0;
    int ay = 0;
    int bx = 0;
    int by = 0;
    exact::rotate_round(p[0], p[1], c, s, ax, ay);
    exact::rotate_round(p[2], p[3], c, s, bx, by);
    const bool bit = d_blur[static_cast<long long>(y + ay) * w + x + ax] <
                     d_blur[static_cast<long long>(y + by) * w + x + bx];
    const unsigned int word = __ballot_sync(kFull, bit);
    if (lane == 0) d_desc[8LL * warp + k] = word;
  }
  if (lane == 0) {
    d_m[2LL * warp] = m10;
    d_m[2LL * warp + 1] = m01;
  }
}

}  // namespace

void resize_bilinear(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst, int dst_w,
                     int dst_h, const int* d_x0, const int* d_x1, const int* d_wx, const int* d_y0,
                     const int* d_y1, const int* d_wy, cudaStream_t s) {
  resize_bilinear_kernel<<<blocks(static_cast<long long>(dst_w) * dst_h), kBlock, 0, s>>>(
      d_src, src_w, d_dst, dst_w, dst_h, d_x0, d_x1, d_wx, d_y0, d_y1, d_wy);
}

void resize_nearest(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst, int dst_w, int dst_h,
                    const int* d_xn, const int* d_yn, cudaStream_t s) {
  resize_nearest_kernel<<<blocks(static_cast<long long>(dst_w) * dst_h), kBlock, 0, s>>>(
      d_src, src_w, d_dst, dst_w, dst_h, d_xn, d_yn);
}

void blur7(const std::uint8_t* d_src, int* d_tmp, std::uint8_t* d_dst, int w, int h,
           cudaStream_t s) {
  const int b = blocks(static_cast<long long>(w) * h);
  blur_h_kernel<<<b, kBlock, 0, s>>>(d_src, d_tmp, w, h);
  blur_v_kernel<<<b, kBlock, 0, s>>>(d_tmp, d_dst, w, h);
}

void fast_scores(const std::uint8_t* d_img, const std::uint8_t* d_mask, std::uint8_t* d_score,
                 int w, int h, int threshold, int border, cudaStream_t s) {
  fast_kernel<<<blocks(static_cast<long long>(w) * h), kBlock, 0, s>>>(d_img, d_mask, d_score, w, h,
                                                                       threshold, border);
}

void nms_compact(const std::uint8_t* d_score, int w, int h, unsigned long long* d_keys,
                 unsigned int* d_count, unsigned int cap, cudaStream_t s) {
  nms_kernel<<<blocks(static_cast<long long>(w) * h), kBlock, 0, s>>>(d_score, w, h, d_keys,
                                                                      d_count, cap);
}

void harris(const std::uint8_t* d_img, int w, const unsigned long long* d_keys, int n,
            long long* d_out, cudaStream_t s) {
  if (n <= 0) return;
  harris_kernel<<<blocks(n), kBlock, 0, s>>>(d_img, w, d_keys, n, d_out);
}

void describe(const std::uint8_t* d_img, const std::uint8_t* d_blur, int w,
              const unsigned int* d_xy, int n, const signed char* d_pattern, long long* d_m,
              unsigned int* d_desc, cudaStream_t s) {
  if (n <= 0) return;
  describe_kernel<<<blocks(32LL * n), kBlock, 0, s>>>(d_img, d_blur, w, d_xy, n, d_pattern, d_m,
                                                      d_desc);
}

}  // namespace sfm::features::cuda
