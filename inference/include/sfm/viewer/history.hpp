#pragma once
// The snapshots of a run in arrival order, for the replay slider (snapshot k = the state after
// the k-th event). Snapshots must arrive with consecutive sequence numbers.

#include <cstddef>
#include <vector>

#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer {

class SnapshotHistory {
 public:
  // Throws std::invalid_argument if s.sequence != size() (a lost or repeated snapshot).
  void push(pipeline::SceneSnapshot s);
  void clear() { items_.clear(); }
  std::size_t size() const { return items_.size(); }
  bool empty() const { return items_.empty(); }
  // k clamped to [0, size - 1]; requires !empty().
  const pipeline::SceneSnapshot& at(std::size_t k) const;
  const pipeline::SceneSnapshot& latest() const { return items_.back(); }

 private:
  std::vector<pipeline::SceneSnapshot> items_;
};

}  // namespace sfm::viewer
