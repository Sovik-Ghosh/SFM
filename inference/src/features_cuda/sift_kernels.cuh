#pragma once
// Kernel launchers of the CUDA SIFT (sift_kernels.cu). Device pointers, row-major float images
// with stride = width; every kernel calls the shared arithmetic of sift_core.hpp. Launch errors
// are checked by the caller.

#include <cstdint>

#include <cuda_runtime.h>

namespace sfm::features::cuda {

// A detected and refined extremum (before the mask and the strongest-N selection).
struct SiftCand {
  int octave;  // absolute octave
  int dog;     // DoG / Gaussian level index (1 .. S)
  int x0;      // detection pixel (restores the CPU's raster detection order)
  int y0;
  int xi;  // integer position after refinement
  int yi;
  float x;  // refined position (octave pixels)
  float y;
  float sigma;
  float response;
};

void to_float(const std::uint8_t* d_gray, float* d_out, int n, cudaStream_t s);
void upsample2(const float* d_src, int src_w, int src_h, float* d_dst, cudaStream_t s);
void downsample2(const float* d_src, int src_w, float* d_dst, int w, int h, cudaStream_t s);
// Separable blur: d_tmp (w * h) holds the horizontal pass. d_taps: 2 r + 1 floats.
void blur(const float* d_src, float* d_tmp, float* d_dst, int w, int h, const float* d_taps, int r,
          cudaStream_t s);
void difference(const float* d_a, const float* d_b, float* d_out, int n, cudaStream_t s);
// Extrema of DoG levels 1 .. S of one octave (d_dog: S + 2 levels of w * h), refined; survivors
// are appended to d_out (atomic counter d_count; entries past `cap` are counted, not written).
void detect(const float* d_dog, int w, int h, int S, int octave, float threshold, float peak,
            float edge, float sigma0, SiftCand* d_out, unsigned int* d_count, unsigned int cap,
            cudaStream_t s);
// Orientations and descriptors of n candidates. Gaussian level of candidate k:
// d_levels + d_level_offset[(octave - first_octave) * (S + 3) + dog], size d_size[2 (octave -
// first_octave)], d_size[... + 1]. Outputs per candidate: d_count[k], d_angle[4 k ..], d_desc[(4 k
// + o) * 32 ..] for o < d_count[k].
void describe(const float* d_levels, const long long* d_level_offset, const int* d_size,
              int first_octave, int S, const SiftCand* d_cand, int n, int max_orientations,
              int* d_count, float* d_angle, unsigned int* d_desc, cudaStream_t s);

}  // namespace sfm::features::cuda
