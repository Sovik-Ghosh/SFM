#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "sfm/matching/verify.hpp"

namespace sfm::matching {

namespace {

class UnionFind {
 public:
  explicit UnionFind(std::size_t n) : parent_(n) { std::iota(parent_.begin(), parent_.end(), 0); }

  std::size_t find(std::size_t a) {
    while (parent_[a] != a) {
      parent_[a] = parent_[parent_[a]];  // path halving
      a = parent_[a];
    }
    return a;
  }

  // The smaller root becomes the parent, so a root is always the smallest node of its set.
  void unite(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (b < a) std::swap(a, b);
    parent_[b] = a;
  }

 private:
  std::vector<std::size_t> parent_;
};

}  // namespace

Tracks build_tracks(const std::vector<PairMatches>& pairs,
                    const std::vector<std::size_t>& keypoint_counts) {
  // Node id = offset[image] + keypoint; node order = (image, keypoint) order.
  std::vector<std::size_t> offset(keypoint_counts.size() + 1, 0);
  for (std::size_t i = 0; i < keypoint_counts.size(); ++i) {
    offset[i + 1] = offset[i] + keypoint_counts[i];
  }
  const std::size_t nodes = offset.back();
  const auto image_of = [&](std::size_t v) {
    return static_cast<int>(std::upper_bound(offset.begin(), offset.end(), v) - offset.begin() - 1);
  };
  for (const PairMatches& p : pairs) {
    if (static_cast<std::size_t>(p.i) >= keypoint_counts.size() ||
        static_cast<std::size_t>(p.j) >= keypoint_counts.size()) {
      throw std::out_of_range("build_tracks: pair image index out of range");
    }
    for (std::size_t k = 0; k < p.size(); ++k) {
      if (static_cast<std::size_t>(p.idx[2 * k]) >=
              keypoint_counts[static_cast<std::size_t>(p.i)] ||
          static_cast<std::size_t>(p.idx[2 * k + 1]) >=
              keypoint_counts[static_cast<std::size_t>(p.j)]) {
        throw std::out_of_range("build_tracks: keypoint index out of range");
      }
    }
  }

  // Strongest pairs first (most inliers; ties by (i, j)), so a wrong match from a weak pair is
  // the one refused when it conflicts.
  std::vector<std::size_t> order(pairs.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (pairs[a].size() != pairs[b].size()) return pairs[a].size() > pairs[b].size();
    return std::make_pair(pairs[a].i, pairs[a].j) < std::make_pair(pairs[b].i, pairs[b].j);
  });

  UnionFind uf(nodes);
  std::vector<std::uint8_t> used(nodes, 0);
  // Images of each component (kept at its root), sorted: a merge is refused when the two
  // components share an image (it would put two keypoints of one image into one track).
  std::vector<std::vector<int>> images(nodes);
  for (const std::size_t pi : order) {
    const PairMatches& p = pairs[pi];
    const std::size_t oi = offset[static_cast<std::size_t>(p.i)];
    const std::size_t oj = offset[static_cast<std::size_t>(p.j)];
    for (std::size_t k = 0; k < p.size(); ++k) {
      const std::size_t a = oi + static_cast<std::size_t>(p.idx[2 * k]);
      const std::size_t b = oj + static_cast<std::size_t>(p.idx[2 * k + 1]);
      for (const std::size_t v : {a, b}) {
        if (used[v] == 0) {
          used[v] = 1;
          images[v] = {image_of(v)};
        }
      }
      std::size_t ra = uf.find(a);
      std::size_t rb = uf.find(b);
      if (ra == rb) continue;
      const auto& ia = images[ra];
      const auto& ib = images[rb];
      std::vector<int> merged;
      merged.reserve(ia.size() + ib.size());
      std::merge(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(merged));
      if (std::adjacent_find(merged.begin(), merged.end()) != merged.end()) continue;  // conflict
      uf.unite(ra, rb);
      const std::size_t r = uf.find(ra);
      images[ra].clear();
      images[rb].clear();
      images[r] = std::move(merged);
    }
  }

  // Group nodes by root. Roots are the smallest node of their set, so visiting nodes in order
  // meets each set's root first: that order fixes the track ids.
  std::vector<std::vector<std::size_t>> members(nodes);
  std::vector<std::size_t> roots;
  for (std::size_t v = 0; v < nodes; ++v) {
    if (used[v] == 0) continue;
    const std::size_t r = uf.find(v);
    if (r == v) roots.push_back(v);
    members[r].push_back(v);
  }
  Tracks t;
  for (const std::size_t r : roots) {
    const auto& m = members[r];  // ascending node id = ascending (image, keypoint)
    if (m.size() < 2) continue;
    for (const std::size_t v : m) {
      const int image = image_of(v);
      t.obs.push_back(t.count);
      t.obs.push_back(image);
      t.obs.push_back(static_cast<int>(v - offset[static_cast<std::size_t>(image)]));
    }
    ++t.count;
  }
  return t;
}

}  // namespace sfm::matching
