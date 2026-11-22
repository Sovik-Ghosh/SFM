#pragma once
// Colour maps for point and camera colouring. Pure functions, no OpenGL.

#include <array>
#include <cstdint>
#include <vector>

namespace sfm::viewer {

using Rgb = std::array<std::uint8_t, 3>;

// Perceptually ordered map (viridis, 5 stops, linear in between) of v in [0, 1] (clamped).
Rgb viridis(float v);

// Value range for colouring: the q_lo and q_hi quantiles of v (robust to outliers). Returns
// {lo, hi} with hi > lo (hi = lo + 1 when all values are equal or v is empty).
std::array<float, 2> robust_range(std::vector<float> v, float q_lo = 0.05F, float q_hi = 0.95F);

// (v - lo) / (hi - lo), clamped to [0, 1].
float normalise(float v, float lo, float hi);

}  // namespace sfm::viewer
