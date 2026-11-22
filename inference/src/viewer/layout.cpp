#include "sfm/viewer/layout.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sfm::viewer {

Layout compute_layout(float width, float height, const PanelSizes& s, float min_view) {
  const float w = std::max(width, 0.0F);
  const float h = std::max(height, 0.0F);
  float header = std::min(s.header, h);
  float timeline = std::min(s.timeline, std::max(h - header, 0.0F));
  float left = s.left;
  float right = s.right;
  const float sides = left + right;
  const float room = std::max(w - min_view, 0.0F);
  if (sides > room) {  // shrink both side panels by the same factor
    const float k = sides > 0 ? room / sides : 0.0F;
    left *= k;
    right *= k;
  }
  const float mid_h = std::max(h - header - timeline, 0.0F);
  Layout l;
  l.header = {0, 0, w, header};
  l.left = {0, header, left, mid_h};
  l.viewport = {left, header, std::max(w - left - right, 0.0F), mid_h};
  l.right = {w - right, header, right, mid_h};
  l.timeline = {0, header + mid_h, w, timeline};
  return l;
}

std::vector<float> mark_positions(int n, float width, float mark) {
  std::vector<float> x;
  if (n <= 0) return x;
  x.reserve(static_cast<std::size_t>(n));
  if (n == 1) {
    x.push_back(0);
    return x;
  }
  const float span = std::max(width - mark, 0.0F);
  for (int i = 0; i < n; ++i) x.push_back(span * static_cast<float>(i) / static_cast<float>(n - 1));
  return x;
}

std::vector<float> stage_fractions(const std::vector<double>& seconds, float min_fraction) {
  const std::size_t n = seconds.size();
  std::vector<float> f(n, 0.0F);
  if (n == 0) return f;
  const double total = std::accumulate(seconds.begin(), seconds.end(), 0.0);
  if (total <= 0) {
    std::fill(f.begin(), f.end(), 1.0F / static_cast<float>(n));
    return f;
  }
  float raised = 0;  // extra given to small stages
  float big = 0;     // total of the stages above the minimum
  for (std::size_t i = 0; i < n; ++i) {
    f[i] = static_cast<float>(seconds[i] / total);
    if (seconds[i] > 0 && f[i] < min_fraction) {
      raised += min_fraction - f[i];
      f[i] = min_fraction;
    } else if (f[i] > min_fraction) {
      big += f[i];
    }
  }
  if (raised > 0 && big > 0) {  // take it from the stages above the minimum, proportionally
    for (float& v : f) {
      if (v > min_fraction) v -= raised * v / big;
    }
  }
  const float sum = std::accumulate(f.begin(), f.end(), 0.0F);
  for (float& v : f) v /= sum;
  return f;
}

int nearest_mark(float x, int n, float width, float mark) {
  if (n <= 0) return -1;
  if (n == 1) return 0;
  const float span = std::max(width - mark, 1e-6F);
  const float t = (x - mark * 0.5F) / span * static_cast<float>(n - 1);
  return std::clamp(static_cast<int>(std::lround(t)), 0, n - 1);
}

}  // namespace sfm::viewer
