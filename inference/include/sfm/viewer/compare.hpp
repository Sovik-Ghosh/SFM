#pragma once
// Comparing two runs in the viewer (no OpenGL): B's reconstruction mapped into A's frame, the
// differences between their configs, and their metrics side by side with the "better" direction.

#include <cstdint>
#include <string>
#include <vector>

#include "sfm/eval/umeyama.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer {

struct SnapshotAlignment {
  bool valid = false;    // false with < 3 images registered in both (or collinear centres)
  eval::Similarity sim;  // X_a = s R X_b + t
  int common = 0;        // images registered in both, matched by dataset id
};

// The similarity that maps B's world frame onto A's: Umeyama on the camera centres of the images
// (matched by dataset id) registered in both.
SnapshotAlignment align_snapshots(const pipeline::SceneSnapshot& b,
                                  const pipeline::SceneSnapshot& a);

// s with its world frame mapped by `sim` (X' = s Q X + u): points move, and a camera x = R X + t
// becomes x = (R Q^T) X' + (s t - R Q^T u), the same image up to scale.
pipeline::SceneSnapshot transform_snapshot(const pipeline::SceneSnapshot& s,
                                           const eval::Similarity& sim);

struct ConfigDifference {
  std::string key;  // dotted path, e.g. "features.sift.peak_threshold"
  std::string a;    // JSON text of the value in A ("" if absent)
  std::string b;
};

// Leaf values that differ between two config JSON texts, sorted by key. run.name and
// run.output_dir are ignored (they always differ between runs).
std::vector<ConfigDifference> config_diff(const std::string& json_a, const std::string& json_b);

enum class Better : std::uint8_t { kLower, kHigher, kNeither };

struct MetricRow {
  std::string name;
  double a = 0;
  double b = 0;
  Better better = Better::kNeither;
  int decimals = 0;
  bool valid = true;  // false when either side lacks the value (e.g. no ground truth)
};

// The rows of the compare table: registered, keypoints / image, verified pairs, points,
// observations, reprojection, rotation and centre errors, BA runs, total time.
std::vector<MetricRow> metric_rows(const pipeline::Metrics& a, const pipeline::Metrics& b,
                                   double seconds_a, double seconds_b);

// +1: B is better, -1: B is worse, 0: equal at the row's precision, no direction, or invalid.
int verdict(const MetricRow& r);

// metrics.json / timings.json text -> structs (fields missing from the text keep their defaults).
pipeline::Metrics parse_metrics(const std::string& json);
pipeline::Timings parse_timings(const std::string& json);

}  // namespace sfm::viewer
