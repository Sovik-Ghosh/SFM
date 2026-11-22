#pragma once
// Kernel launchers of the CUDA ORB (orb_kernels.cu). All take device pointers and a stream; images
// are row-major with stride = width. Launch errors are checked by the caller
// (CHECK(cudaGetLastError())).

#include <cstdint>

#include <cuda_runtime.h>

namespace sfm::features::cuda {

// dst(x, y) from src via the axis tables (exact::bilinear); tables have dst_w / dst_h entries.
void resize_bilinear(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst, int dst_w,
                     int dst_h, const int* d_x0, const int* d_x1, const int* d_wx, const int* d_y0,
                     const int* d_y1, const int* d_wy, cudaStream_t s);
void resize_nearest(const std::uint8_t* d_src, int src_w, std::uint8_t* d_dst, int dst_w, int dst_h,
                    const int* d_xn, const int* d_yn, cudaStream_t s);

// Separable integer 7x7 Gaussian (exact::blur_weight); d_tmp holds w * h ints.
void blur7(const std::uint8_t* d_src, int* d_tmp, std::uint8_t* d_dst, int w, int h,
           cudaStream_t s);

// FAST score map: score where score > threshold, inside the border and the mask (nullptr = none);
// 0 elsewhere.
void fast_scores(const std::uint8_t* d_img, const std::uint8_t* d_mask, std::uint8_t* d_score,
                 int w, int h, int threshold, int border, cudaStream_t s);

// 3x3 NMS with the raster tie rule; every kept pixel appends the sort key
// (255 - score) << 40 | y << 20 | x to d_keys (atomic counter d_count, capacity `cap`).
void nms_compact(const std::uint8_t* d_score, int w, int h, unsigned long long* d_keys,
                 unsigned int* d_count, unsigned int cap, cudaStream_t s);

// Harris (exact::harris_int) of n candidates given by their sort keys.
void harris(const std::uint8_t* d_img, int w, const unsigned long long* d_keys, int n,
            long long* d_out, cudaStream_t s);

// Per keypoint (x, y packed as y << 20 | x): moments on d_img, steered BRIEF on d_blur.
// d_pattern: 256 x 4 int8 offsets. Outputs: d_m (2 per keypoint), d_desc (8 words per keypoint).
void describe(const std::uint8_t* d_img, const std::uint8_t* d_blur, int w,
              const unsigned int* d_xy, int n, const signed char* d_pattern, long long* d_m,
              unsigned int* d_desc, cudaStream_t s);

}  // namespace sfm::features::cuda
