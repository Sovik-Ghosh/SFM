#pragma once
// FAST-9/16 corner detection, 3x3 non-maximum suppression and the Harris corner response.
// A pixel c is a FAST corner when 9 contiguous pixels of the radius-3 Bresenham circle are all
// brighter than c + t or all darker than c - t.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sfm/data/image_io.hpp"

namespace sfm::features {

struct Corner {
  int x = 0;
  int y = 0;
  int score = 0;  // FAST score (see fast_score)
};

// The largest t for which (x, y) is still a corner: max over the 16 arcs of 9 pixels of the
// smallest |circle - centre| on the arc (bright and dark arcs both counted). The point is a corner
// for threshold t exactly when fast_score > t. Requires 3 <= x < W - 3 and 3 <= y < H - 3.
int fast_score(const data::Image8& img, int x, int y);

// Score map (W * H, 0 where not a corner) -> corners that are local maxima of their 3x3
// neighbourhood. Ties are broken by raster order (the first pixel wins), so a plateau keeps one.
std::vector<Corner> nms_3x3(const std::vector<int>& scores, int width, int height);

// FAST corners with score > threshold, at least `border` (>= 3) pixels from every edge, where the
// mask (if given, same size) is non-zero, after 3x3 non-maximum suppression. Raster order.
std::vector<Corner> detect_fast(const data::Image8& img, const data::Image8* mask, int threshold,
                                int border);

// Harris response det(M) - k trace(M)^2 of the 7x7 window of Sobel gradients centred on (x, y),
// k = 0.04: harris_int is 25 x that, exact in int64 (the ranking key); harris_response scales it
// like OpenCV's ORB. Require the window plus the Sobel ring inside the image.
std::int64_t harris_int(const data::Image8& img, int x, int y);
float harris_response(const data::Image8& img, int x, int y);

}  // namespace sfm::features
