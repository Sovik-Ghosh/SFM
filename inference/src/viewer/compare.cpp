#include "sfm/viewer/compare.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "sfm/core/error.hpp"

namespace sfm::viewer {

namespace {

Eigen::Vector3d centre(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  return -R.transpose() * t;
}

void flatten(const nlohmann::json& j, const std::string& prefix,
             std::map<std::string, std::string>& out) {
  if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it) {
      flatten(it.value(), prefix.empty() ? it.key() : prefix + "." + it.key(), out);
    }
  } else {
    out[prefix] = j.dump();
  }
}

}  // namespace

SnapshotAlignment align_snapshots(const pipeline::SceneSnapshot& b,
                                  const pipeline::SceneSnapshot& a) {
  SnapshotAlignment r;
  std::map<int, std::size_t> in_a;
  for (std::size_t i = 0; i < a.ids.size(); ++i) {
    if (i < a.registered.size() && a.registered[i] != 0) in_a[a.ids[i]] = i;
  }
  std::vector<Eigen::Vector3d> src;
  std::vector<Eigen::Vector3d> dst;
  for (std::size_t j = 0; j < b.ids.size(); ++j) {
    if (j >= b.registered.size() || b.registered[j] == 0) continue;
    const auto it = in_a.find(b.ids[j]);
    if (it == in_a.end()) continue;
    src.push_back(centre(b.R[j], b.t[j]));
    dst.push_back(centre(a.R[it->second], a.t[it->second]));
  }
  r.common = static_cast<int>(src.size());
  if (src.size() < 3) return r;
  Eigen::Matrix3Xd s(3, static_cast<Eigen::Index>(src.size()));
  Eigen::Matrix3Xd d(3, static_cast<Eigen::Index>(dst.size()));
  for (std::size_t k = 0; k < src.size(); ++k) {
    s.col(static_cast<Eigen::Index>(k)) = src[k];
    d.col(static_cast<Eigen::Index>(k)) = dst[k];
  }
  try {
    r.sim = eval::umeyama(s, d);
    r.valid = true;
  } catch (const GeometryError&) {
    r.valid = false;  // collinear centres
  }
  return r;
}

pipeline::SceneSnapshot transform_snapshot(const pipeline::SceneSnapshot& s,
                                           const eval::Similarity& sim) {
  pipeline::SceneSnapshot o = s;
  for (auto& p : o.points) p = sim.apply(p);
  for (std::size_t i = 0; i < o.R.size(); ++i) {
    const Eigen::Matrix3d r = s.R[i] * sim.R.transpose();
    o.t[i] = sim.s * s.t[i] - r * sim.t;
    o.R[i] = r;
  }
  return o;
}

std::vector<ConfigDifference> config_diff(const std::string& json_a, const std::string& json_b) {
  std::map<std::string, std::string> fa;
  std::map<std::string, std::string> fb;
  flatten(nlohmann::json::parse(json_a), "", fa);
  flatten(nlohmann::json::parse(json_b), "", fb);
  std::set<std::string> keys;
  for (const auto& [k, v] : fa) keys.insert(k);
  for (const auto& [k, v] : fb) keys.insert(k);
  std::vector<ConfigDifference> out;
  for (const std::string& k : keys) {
    if (k == "run.name" || k == "run.output_dir") continue;
    const auto ia = fa.find(k);
    const auto ib = fb.find(k);
    const std::string va = ia == fa.end() ? "" : ia->second;
    const std::string vb = ib == fb.end() ? "" : ib->second;
    if (va != vb) out.push_back({k, va, vb});
  }
  return out;
}

std::vector<MetricRow> metric_rows(const pipeline::Metrics& a, const pipeline::Metrics& b,
                                   double seconds_a, double seconds_b) {
  auto row = [](std::string name, double va, double vb, Better better, int decimals,
                bool valid = true) {
    MetricRow r;
    r.name = std::move(name);
    r.a = va;
    r.b = vb;
    r.better = better;
    r.decimals = decimals;
    r.valid = valid;
    return r;
  };
  const bool gt = a.rotation_median_deg >= 0 && b.rotation_median_deg >= 0;
  return {
      row("Registered", static_cast<double>(a.registered_ids.size()),
          static_cast<double>(b.registered_ids.size()), Better::kHigher, 0),
      row("Keypoints / image", a.keypoints_mean, b.keypoints_mean, Better::kHigher, 0),
      row("Verified pairs", a.verified_pairs, b.verified_pairs, Better::kHigher, 0),
      row("Points", a.points, b.points, Better::kHigher, 0),
      row("Observations", a.observations, b.observations, Better::kHigher, 0),
      row("Reprojection px", a.reproj_median_px, b.reproj_median_px, Better::kLower, 3),
      row("Rotation median °", a.rotation_median_deg, b.rotation_median_deg, Better::kLower, 3, gt),
      row("Rotation max °", a.rotation_max_deg, b.rotation_max_deg, Better::kLower, 3, gt),
      row("Centre median %", a.centre_median_pct, b.centre_median_pct, Better::kLower, 3, gt),
      row("Centre max %", a.centre_max_pct, b.centre_max_pct, Better::kLower, 3, gt),
      row("BA runs", a.ba_runs, b.ba_runs, Better::kNeither, 0),
      row("Total s", seconds_a, seconds_b, Better::kLower, 2, seconds_a > 0 && seconds_b > 0),
  };
}

int verdict(const MetricRow& r) {
  if (!r.valid || r.better == Better::kNeither) return 0;
  const double scale = std::pow(10.0, r.decimals);
  const double da = std::round(r.a * scale);
  const double db = std::round(r.b * scale);
  if (da == db) return 0;
  const bool b_higher = db > da;
  return (r.better == Better::kHigher) == b_higher ? 1 : -1;
}

pipeline::Metrics parse_metrics(const std::string& json) {
  pipeline::Metrics m;
  const auto j = nlohmann::json::parse(json);
  m.images = j.value("images", 0);
  m.keypoints_mean = j.value("keypoints_mean", 0.0);
  m.verified_pairs = j.value("verified_pairs", 0);
  m.tracks = j.value("tracks", 0);
  m.points = j.value("points", 0);
  m.observations = j.value("observations", 0);
  m.pnp_failures = j.value("pnp_failures", 0);
  m.ba_runs = j.value("ba_runs", 0);
  if (j.contains("registered_ids")) m.registered_ids = j["registered_ids"].get<std::vector<int>>();
  if (j.contains("reprojection_px")) {
    const auto& r = j["reprojection_px"];
    m.reproj_median_px = r.value("median", 0.0);
    m.reproj_mean_px = r.value("mean", 0.0);
    m.reproj_p95_px = r.value("p95", 0.0);
  }
  if (j.contains("rotation_error_deg")) {
    m.rotation_median_deg = j["rotation_error_deg"].value("median", -1.0);
    m.rotation_max_deg = j["rotation_error_deg"].value("max", -1.0);
  }
  if (j.contains("centre_error_pct")) {
    m.centre_median_pct = j["centre_error_pct"].value("median", -1.0);
    m.centre_max_pct = j["centre_error_pct"].value("max", -1.0);
  }
  return m;
}

pipeline::Timings parse_timings(const std::string& json) {
  pipeline::Timings t;
  const auto j = nlohmann::json::parse(json);
  t.load = j.value("load", 0.0);
  t.features = j.value("features", 0.0);
  t.matching = j.value("matching", 0.0);
  t.tracks = j.value("tracks", 0.0);
  t.reconstruction = j.value("reconstruction", 0.0);
  t.ba = j.value("ba", 0.0);
  t.export_ = j.value("export", 0.0);
  t.total = j.value("total", 0.0);
  return t;
}

}  // namespace sfm::viewer
