#include "sfm/viewer/history.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace sfm::viewer {

void SnapshotHistory::push(pipeline::SceneSnapshot s) {
  if (s.sequence != static_cast<int>(items_.size())) {
    throw std::invalid_argument("SnapshotHistory: expected sequence " +
                                std::to_string(items_.size()) + ", got " +
                                std::to_string(s.sequence));
  }
  items_.push_back(std::move(s));
}

const pipeline::SceneSnapshot& SnapshotHistory::at(std::size_t k) const {
  if (items_.empty()) throw std::out_of_range("SnapshotHistory::at: empty");
  return items_[std::min(k, items_.size() - 1)];
}

}  // namespace sfm::viewer
