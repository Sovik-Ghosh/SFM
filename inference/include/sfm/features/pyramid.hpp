#pragma once
// Image pyramid: level l has size round(W / scale^l) x round(H / scale^l). Gray levels use
// bilinear interpolation with pixel-centre alignment, masks use nearest neighbour (stays binary).
// All arithmetic is integer (sfm/features/exact.hpp), driven by per-axis tables that the CUDA
// version shares, so CPU and GPU pyramids are byte-identical.

#include <cstdint>
#include <vector>

#include "sfm/data/image_io.hpp"

namespace sfm::features {

enum class Resample : std::uint8_t { kBilinear, kNearest };

// Sampling table of one axis (src -> dst pixels). Output pixel x samples the source at
// f = clamp((x + 0.5) * src / dst - 0.5, 0, src - 1): bilinear between i0 = floor(f) and
// i1 = min(i0 + 1, src - 1) with weight w = round((f - i0) * 2048) on i1; nearest takes
// n = min(floor((x + 0.5) * src / dst), src - 1).
struct AxisTable {
  std::vector<int> i0;
  std::vector<int> i1;
  std::vector<int> w;
  std::vector<int> nearest;
};
AxisTable axis_table(int src, int dst);

// Resize with the axis tables (see AxisTable). Throws std::invalid_argument on a size < 1.
data::Image8 resize(const data::Image8& src, int width, int height, Resample method);

// levels >= 1, scale > 1. Level 0 is a copy of `base`; level l is resized from level l - 1 (like
// OpenCV's ORB, which makes the higher levels smoother than a direct resize of the base: measured
// on bunny 0012, keypoints within 1 px of OpenCV's at levels 2-5 rise from 59-80 % to 81-93 %).
// Level sizes are computed from the base size, so rounding does not accumulate.
std::vector<data::Image8> build_pyramid(const data::Image8& base, int levels, double scale,
                                        Resample method = Resample::kBilinear);

// 7x7 Gaussian blur, sigma 2, border replicated (smooths a level before the descriptor tests).
// Integer weights (exact::blur_weight, sum 256 per axis): horizontal sums kept exact, one rounding
// at the end: (sum + 2^15) >> 16.
data::Image8 gaussian_blur7(const data::Image8& src);

}  // namespace sfm::features
