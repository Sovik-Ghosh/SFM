#pragma once
// SIFT (Lowe, IJCV 2004) in the VLFeat variant that COLMAP uses on the CPU:
//   1. float image / 255; first octave -1 = image upsampled x2 (VLFeat's copy-and-upsample)
//   2. per octave S + 3 Gaussian levels (s = -1 .. S + 1, sigma(s) = sigma0 2^(s / S), the input
//      assumed blurred by 0.5 px), separable blur with host-computed taps (radius ceil(4 sigma)),
//      replicated borders; next octave = every 2nd pixel of level S - 1
//   3. DoG; strict 3x3x3 extrema with |D| >= 0.8 peak_threshold on levels s = 0 .. S - 1
//   4. sub-pixel refinement (<= 5 steps, VLFeat's rules), then |D(x^)| > peak_threshold, edge
//      ratio (tr H)^2 / det H < (r + 1)^2 / r, offsets < 1.5
//   5. keep the max_features strongest locations (|D(x^)|), ties by (octave, level, y, x)
//   6. up to max_orientations orientations: 36-bin gradient histogram (radius 3 * 1.5 sigma,
//      Gaussian weight), 6 circular box smoothings, peaks >= 0.8 max, parabolic interpolation
//   7. descriptor 4 x 4 x 8, trilinear binning, Gaussian window (sigma = 2 bins), normalise,
//      clamp 0.2, renormalise, RootSIFT (L1, sqrt), x 512 rounded, clamped to 255 -> u8
// Positions are level-0 pixel-centre coordinates (pixel (0, 0) at (0, 0)); COLMAP writes +0.5.
// Every float operation is a correctly rounded +, -, *, /, sqrt, floor or an exact_math call, in a
// fixed order (sift_core.hpp), so the CUDA version (sift_cuda.hpp) produces identical bytes.

#include <cstdint>
#include <vector>

#include "sfm/data/image_io.hpp"
#include "sfm/features/features.hpp"

namespace sfm::features {

struct SiftParams {
  std::uint32_t max_features = 8192;  // locations (before orientations)
  int octaves = 4;
  int octave_resolution = 3;            // S
  int first_octave = -1;                // -1: start from the image upsampled x2
  float peak_threshold = 0.02F / 3.0F;  // COLMAP's default (0.02 / S)
  float edge_threshold = 10.0F;
  int max_orientations = 2;
};

// Separable Gaussian taps (normalised, radius ceil(4 sigma), at least 1), computed in double and
// rounded to float: the one place transcendental functions of the host library are used; the
// GPU receives the same array.
std::vector<float> gaussian_taps(double sigma);

// The blur steps of one octave: entry 0 = blur of the first level (from the upsampled /
// downsampled image; empty if none is needed), entry l = blur from level l - 1 to level l.
std::vector<std::vector<float>> octave_blur_taps(const SiftParams& p, int octave);

// Internals shared with the CUDA version (host side).
namespace sift_detail {

// One refined location (before orientations). Positions in octave pixels.
struct Candidate {
  int octave = 0;  // absolute octave (first_octave ..)
  int dog = 0;     // DoG / Gaussian level index of the detection (s - s_min, 1 .. S)
  int xi = 0;      // integer position after refinement
  int yi = 0;
  float x = 0.0F;  // refined position
  float y = 0.0F;
  float sigma = 0.0F;     // keypoint scale (octave pixels)
  float response = 0.0F;  // |D(x^)|
  std::uint32_t seq = 0;  // detection order (last tie-break)
};

// sigma0 = 1.6 2^(1 / S) (VLFeat), as float.
float sigma0(int octave_resolution);
// Size of octave o of a w x h image (o < 0: upsampled); false if it is too small (< 8 px).
bool octave_size(int width, int height, int octave, int& w, int& h);
// Keeps the max_features strongest candidates: response descending, ties by (octave, dog, yi,
// xi, seq).
void select_candidates(std::vector<Candidate>& c, std::uint32_t max_features);
// Mask test at the rounded level-0 position (mask: nullptr = everything allowed).
bool mask_allows(const data::Image8* mask, const Candidate& c);
// Appends one keypoint (orientation in radians, descriptor of kSiftWords words) to out.
void append_keypoint(Features& out, const Candidate& c, float angle_rad, const std::uint32_t* desc,
                     int first_octave);

}  // namespace sift_detail

// Throws std::invalid_argument for parameters the detector does not support.
void check_sift_params(const SiftParams& p);

// The whole detector. `mask` (optional, same size as gray, non-zero = allowed) restricts keypoint
// positions (rounded level-0 position). Output: Features with words = kSiftWords; angle in
// degrees [0, 360), response = |D(x^)|, octave = octave - first_octave; ordered by location
// rank (step 5), the orientations of one location consecutive in histogram order.
Features detect_sift(const data::Image8& gray, const data::Image8* mask, const SiftParams& p);

}  // namespace sfm::features
