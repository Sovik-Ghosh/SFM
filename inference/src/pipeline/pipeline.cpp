#include "sfm/pipeline/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>

#include <Eigen/LU>
#include <nlohmann/json.hpp>

#include "sfm/core/error.hpp"
#include "sfm/core/npy.hpp"
#include "sfm/core/parallel.hpp"
#include "sfm/core/trace.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/image_list.hpp"
#include "sfm/data/krt.hpp"
#include "sfm/data/rig.hpp"
#include "sfm/eval/pose_error.hpp"
#include "sfm/export/export.hpp"
#include "sfm/matching/verify.hpp"
#include "sfm/pipeline/stage_io.hpp"
#include "sfm/recon/distortion.hpp"
#include "sfm/recon/incremental.hpp"

namespace sfm::pipeline {

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void write_text(const fs::path& file, const std::string& text) {
  std::ofstream out(file);
  if (!out) throw IoError("cannot write " + file.string());
  out << text;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const auto k = static_cast<std::size_t>(std::lround(q * static_cast<double>(v.size() - 1)));
  return v[std::min(k, v.size() - 1)];
}

Metrics compute_metrics(const recon::Reconstruction& rec, const recon::TrackIndex& index,
                        const std::vector<features::Features>& f, std::size_t pairs, int tracks,
                        const std::vector<int>& ids, const std::vector<data::Krt>& gt,
                        const recon::IncrementalReport& rep) {
  Metrics m;
  m.images = static_cast<int>(f.size());
  double kp = 0.0;
  for (const auto& x : f) kp += static_cast<double>(x.size());
  m.keypoints_mean = f.empty() ? 0.0 : kp / static_cast<double>(f.size());
  m.verified_pairs = static_cast<int>(pairs);
  m.tracks = tracks;
  if (rec.order.size() >= 2) {
    m.init_i = ids[static_cast<std::size_t>(rec.order[0])];
    m.init_j = ids[static_cast<std::size_t>(rec.order[1])];
  }
  std::vector<eval::Pose> est;
  std::vector<eval::Pose> tru;
  for (std::size_t i = 0; i < rec.cameras.size(); ++i) {
    if (!rec.cameras[i].registered) continue;
    m.registered_ids.push_back(ids[i]);
    est.push_back(rec.cameras[i].pose);
    tru.push_back({gt[i].R, gt[i].t});
  }
  m.points = rec.point_count();
  const auto e = recon::reprojection_errors(rec, index);
  m.observations = static_cast<int>(e.size());
  if (!e.empty()) {
    m.reproj_mean_px = std::accumulate(e.begin(), e.end(), 0.0) / static_cast<double>(e.size());
    m.reproj_median_px = percentile(e, 0.5);
    m.reproj_p95_px = percentile(e, 0.95);
  }
  if (est.size() >= 3) {
    // Calibration with intrinsics only (P = K [I | 0] for every image: no ground-truth poses)
    // puts every true centre at the origin, so the alignment is undetermined: no pose errors (-1).
    try {
      const auto pe = eval::pose_error(est, tru);
      m.rotation_median_deg = percentile(pe.rotation_deg, 0.5);
      m.rotation_max_deg = pe.max_rotation_deg();
      m.centre_median_pct = percentile(pe.centre_pct, 0.5);
      m.centre_max_pct = pe.max_centre_pct();
    } catch (const GeometryError&) {
      m.rotation_median_deg = -1.0;  // none, as without registered cameras
      m.rotation_max_deg = -1.0;
      m.centre_median_pct = -1.0;
      m.centre_max_pct = -1.0;
    }
  }
  m.pnp_failures = rep.pnp_failures;
  m.ba_runs = rep.ba_runs;
  return m;
}

}  // namespace

namespace {

struct FocalSearch {
  double scale = 1.0;
  double cost = 0.0;
  int evaluations = 0;
};

// One scale factor for every camera's focal length (variable projection: for each candidate the
// final BA re-fits poses and points from the same start, and its cost is the candidate's score).
// Golden-section search on log(scale) in [log 0.74, log 1.35]; the cost is in pixels, so it
// compares across focal lengths. rec gets the best candidate's result.
FocalSearch refine_focal(recon::Reconstruction& rec, const recon::TrackIndex& index,
                         const config::BaConfig& fc, const BaMethod& ba_method) {
  const recon::Reconstruction start = rec;
  FocalSearch best;
  recon::Reconstruction best_rec = rec;
  const auto evaluate = [&](double log_s) {
    recon::Reconstruction r = start;
    const double s = std::exp(log_s);
    for (auto& c : r.cameras) {
      c.K(0, 0) *= s;
      c.K(1, 1) *= s;
      c.K(0, 1) *= s;  // skew scales with the focal length
    }
    const ba::Report rep = ba_method(r, index, fc, {});
    ++best.evaluations;
    if (best.evaluations == 1 || rep.final_cost < best.cost) {
      best.cost = rep.final_cost;
      best.scale = s;
      best_rec = std::move(r);
    }
    return rep.final_cost;
  };
  constexpr double kGolden = 0.6180339887498949;
  double a = std::log(0.74);
  double b = std::log(1.35);
  evaluate(0.0);  // the current focal length: kept unless a candidate is better
  double x1 = b - kGolden * (b - a);
  double x2 = a + kGolden * (b - a);
  double f1 = evaluate(x1);
  double f2 = evaluate(x2);
  while (b - a > 2e-3) {  // scale resolution ~0.2 %
    if (f1 < f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = b - kGolden * (b - a);
      f1 = evaluate(x1);
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + kGolden * (b - a);
      f2 = evaluate(x2);
    }
  }
  rec = std::move(best_rec);
  return best;
}

// Robust reprojection cost measured where the keypoints were recorded: project, distort, compare
// with the recorded (raw) position. Comparable across distortion candidates, unlike a cost in
// undistorted pixels (which grows or shrinks with the undistortion itself).
double recorded_cost(const recon::Reconstruction& rec, const recon::TrackIndex& index,
                     const std::vector<features::Features>& raw, const recon::Radial& d,
                     double huber_px) {
  double cost = 0.0;
  for (int t = 0; t < index.track_count(); ++t) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    const Eigen::Vector3d& X = rec.points[static_cast<std::size_t>(t)];
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      if (!recon::observation_used(rec, index, o)) continue;
      const auto& c = rec.cameras[static_cast<std::size_t>(o->image)];
      const Eigen::Vector3d p = c.K * (c.pose.R * X + c.pose.t);
      if (p.z() <= 0.0) continue;
      const Eigen::Vector2d ideal = p.head<2>() / p.z();
      const auto& xy = raw[static_cast<std::size_t>(o->image)].xy;
      const auto k = 2 * static_cast<std::size_t>(o->keypoint);
      const double e = (recon::distort(c.K, d, ideal) - Eigen::Vector2d(xy[k], xy[k + 1])).norm();
      cost += e <= huber_px ? e * e : 2.0 * huber_px * e - huber_px * huber_px;
    }
  }
  return 0.5 * cost;
}

// How far from the centre the keypoints reach: 90th percentile of the normalised r^2 of every
// recorded keypoint. Radial distortion shows only where r^2 is large.
constexpr double kMinDistortionCoverage = 0.2;

double keypoint_coverage(const std::vector<features::Features>& raw,
                         const std::vector<Eigen::Matrix3d>& K) {
  std::vector<double> r2;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const Eigen::Matrix3d Ki = K[i].inverse();
    for (std::size_t k = 0; k + 1 < raw[i].xy.size(); k += 2) {
      const Eigen::Vector3d x = Ki * Eigen::Vector3d(raw[i].xy[k], raw[i].xy[k + 1], 1.0);
      r2.push_back(x.head<2>().squaredNorm() / (x.z() * x.z()));
    }
  }
  return percentile(r2, 0.9);
}

struct DistortionSearch {
  double k1 = 0.0;
  double cost = 0.0;
  double cost_start = 0.0;
  int evaluations = 0;
  std::vector<features::Features> feats;  // raw keypoints undistorted with the chosen k1
};

// Radial k1 by golden-section search in [-0.4, 0.4]: each candidate undistorts the recorded
// keypoints with the cameras' current K, re-runs the final BA from the same start and is scored
// by recorded_cost. rec gets the best candidate's result. k2 is kept.
DistortionSearch refine_distortion(recon::Reconstruction& rec, const matching::Tracks& tracks,
                                   const std::vector<features::Features>& raw, double k1_start,
                                   double k2, const config::BaConfig& fc,
                                   const BaMethod& ba_method) {
  const recon::Reconstruction start = rec;
  std::vector<Eigen::Matrix3d> K;
  K.reserve(start.cameras.size());
  for (const auto& c : start.cameras) K.push_back(c.K);
  DistortionSearch best;
  recon::Reconstruction best_rec = rec;
  const auto evaluate = [&](double k1) {
    const recon::Radial d{k1, k2};
    std::vector<features::Features> f = recon::undistort_features(raw, K, d);
    const recon::TrackIndex idx(tracks, f);
    recon::Reconstruction r = start;
    ba_method(r, idx, fc, {});
    const double cost = recorded_cost(r, idx, raw, d, fc.huber_px);
    ++best.evaluations;
    if (best.evaluations == 1 || cost < best.cost) {
      best.cost = cost;
      best.k1 = k1;
      best.feats = std::move(f);
      best_rec = std::move(r);
    }
    return cost;
  };
  best.cost_start = evaluate(k1_start);  // the current k1: kept unless a candidate is better
  constexpr double kGolden = 0.6180339887498949;
  double a = -0.4;
  double b = 0.4;
  double x1 = b - kGolden * (b - a);
  double x2 = a + kGolden * (b - a);
  double f1 = evaluate(x1);
  double f2 = evaluate(x2);
  while (b - a > 1e-3) {
    if (f1 < f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = b - kGolden * (b - a);
      f1 = evaluate(x1);
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + kGolden * (b - a);
      f2 = evaluate(x2);
    }
  }
  rec = std::move(best_rec);
  return best;
}

}  // namespace

// Gray value per track: the pixel under its first observation.
std::vector<std::uint8_t> track_gray(const recon::TrackIndex& index,
                                     const std::vector<data::Image8>& gray) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(index.track_count()), 128);
  for (int t = 0; t < index.track_count(); ++t) {
    const auto* o = index.begin(t);
    const auto& img = gray[static_cast<std::size_t>(o->image)];
    const int x = std::clamp(static_cast<int>(std::lround(o->uv.x())), 0, img.width - 1);
    const int y = std::clamp(static_cast<int>(std::lround(o->uv.y())), 0, img.height - 1);
    out[static_cast<std::size_t>(t)] =
        img.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) +
                   static_cast<std::size_t>(x)];
  }
  return out;
}

std::vector<exporter::Rgb> track_rgb(const recon::TrackIndex& index,
                                     const std::vector<fs::path>& images) {
  std::vector<exporter::Rgb> out(static_cast<std::size_t>(index.track_count()),
                                 exporter::Rgb{128, 128, 128});
  std::vector<std::vector<int>> by_image(images.size());  // tracks whose first observation is here
  for (int t = 0; t < index.track_count(); ++t) {
    by_image[static_cast<std::size_t>(index.begin(t)->image)].push_back(t);
  }
  for (std::size_t i = 0; i < images.size(); ++i) {
    if (by_image[i].empty()) continue;
    const data::ImageRgb img = data::load_rgb(images[i]);
    for (const int t : by_image[i]) {
      const auto* o = index.begin(t);
      const int x = std::clamp(static_cast<int>(std::lround(o->uv.x())), 0, img.width - 1);
      const int y = std::clamp(static_cast<int>(std::lround(o->uv.y())), 0, img.height - 1);
      const auto row = static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width);
      const std::size_t k = 3 * (row + static_cast<std::size_t>(x));
      out[static_cast<std::size_t>(t)] = {img.rgb[k], img.rgb[k + 1], img.rgb[k + 2]};
    }
  }
  return out;
}

SceneSnapshot make_snapshot(const recon::Reconstruction& rec, const recon::TrackIndex& index,
                            const std::vector<int>& ids, const std::vector<std::uint8_t>& gray,
                            int sequence) {
  SceneSnapshot s;
  s.sequence = sequence;
  s.ids = ids;
  for (const auto& c : rec.cameras) {
    s.K.push_back(c.K);
    s.R.push_back(c.pose.R);
    s.t.push_back(c.pose.t);
    s.registered.push_back(c.registered ? 1 : 0);
  }
  s.order = rec.order;
  for (int t = 0; t < index.track_count(); ++t) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    const Eigen::Vector3d& X = rec.points[static_cast<std::size_t>(t)];
    double err = 0.0;
    int n = 0;
    for (const auto* o = index.begin(t); o != index.end(t); ++o) {
      if (!recon::observation_used(rec, index, o)) continue;
      const auto& c = rec.cameras[static_cast<std::size_t>(o->image)];
      err += recon::reprojection_error(c.K, c.pose, X, o->uv);
      ++n;
    }
    s.points.push_back(X);
    s.point_gray.push_back(gray[static_cast<std::size_t>(t)]);
    s.point_error_px.push_back(n > 0 ? static_cast<float>(err / n) : 0.0F);
    s.point_track_length.push_back(n);
  }
  return s;
}

Stage parse_stage(const std::string& name) {
  for (Stage s : {Stage::kFeatures, Stage::kMatching, Stage::kTracks, Stage::kReconstruction,
                  Stage::kExport}) {
    if (to_string(s) == name) return s;
  }
  throw ConfigError("stage: unknown '" + name +
                    "' (features|matching|tracks|reconstruction|export)");
}

std::string to_string(Stage s) {
  switch (s) {
    case Stage::kFeatures:
      return "features";
    case Stage::kMatching:
      return "matching";
    case Stage::kTracks:
      return "tracks";
    case Stage::kReconstruction:
      return "reconstruction";
    case Stage::kExport:
      return "export";
  }
  return "?";
}

namespace {

RunResult run_once(const config::Config& cfg_in, const RunOptions& opt, core::Logger& log,
                   const Registry& registry, ProgressSink* sink) {
  if (opt.to < opt.from) throw ConfigError("--to must not come before --from");
  const auto t_total = Clock::now();
  const bool any_auto = cfg_in.features.method == "auto" || cfg_in.matching.method == "auto" ||
                        cfg_in.ba.method == "auto";
  const auto is_cuda = [](const std::string& m) {
    return m.size() > 5 && m.compare(m.size() - 5, 5, "_cuda") == 0;
  };
  const bool any_cuda = is_cuda(cfg_in.features.method) || is_cuda(cfg_in.matching.method) ||
                        is_cuda(cfg_in.ba.method);
  const Backend backend = any_auto || any_cuda ? detect_backend() : Backend{};
  // A config that asks for CUDA methods runs on their CPU twins when this build has no CUDA or
  // no GPU is visible (logged as run.backend_fallback), so every config runs everywhere.
  const bool fallback = any_cuda && !backend.gpu;
  const config::Config cfg = fallback   ? use_backend(cfg_in, false)
                             : any_auto ? resolve_auto_methods(cfg_in, backend.gpu)
                                        : cfg_in;
  const auto check_cancel = [&] {
    if (sink != nullptr && sink->cancelled()) throw Cancelled("run cancelled");
  };
  std::string range_name;  // NVTX copies the name on push
  const auto begin = [&](Stage s) {
    check_cancel();
    range_name = "stage." + to_string(s);
    core::trace_push(range_name.c_str());
    if (sink != nullptr) sink->stage_begin(s);
  };
  const auto end = [&](Stage s, double seconds) {
    core::trace_pop();
    if (sink != nullptr) sink->stage_end(s, seconds);
  };
  RunResult res;
  res.run_dir = cfg.run.output_dir / cfg.run.name;
  const fs::path stages = res.run_dir / "stages";
  fs::create_directories(res.run_dir);
  log.add_file(res.run_dir / "run.log");
  write_text(res.run_dir / "config.json", config::to_json(cfg));
  log.info("run.start",
           {{"run", cfg.run.name}, {"from", to_string(opt.from)}, {"to", to_string(opt.to)}});
  if (fallback) {
    log.warn("run.backend_fallback",
             {{"reason", !built_with_cuda()          ? std::string("built without CUDA")
                         : !backend.unusable.empty() ? backend.unusable
                                                     : std::string("no CUDA device visible")},
              {"features", cfg.features.method},
              {"matching", cfg.matching.method},
              {"ba", cfg.ba.method}});
  }
  if (any_auto) {
    log.info("run.backend", {{"gpu", backend.gpu},
                             {"device", backend.device.empty() ? "none" : backend.device},
                             {"features", cfg.features.method},
                             {"matching", cfg.matching.method},
                             {"ba", cfg.ba.method}});
  }

  // Methods are looked up first, so a typo fails before any work.
  const FeatureMethod& feature_method = registry.features(cfg.features.method);
  const MatchMethod& match_method = registry.matching(cfg.matching.method);
  const BaMethod& ba_method = registry.ba(cfg.ba.method);

  // --- images and calibration --------------------------------------------------------------
  auto t0 = Clock::now();
  const config::DatasetConfig& d = cfg.dataset;
  std::vector<int> ids;
  std::vector<std::string> names;
  std::vector<data::Krt> gt;
  std::vector<Eigen::Matrix3d> K;
  std::vector<data::Image8> gray;
  std::vector<data::Image8> masks;
  const auto files = data::list_images(d);
  if (cfg.memory.max_images < files.size()) {
    throw ConfigError("memory.max_images: " + std::to_string(cfg.memory.max_images) +
                      " is less than the " + std::to_string(files.size()) + " images in the range");
  }
  int rotated = 0;
  for (const data::ImageEntry& e : files) {
    ids.push_back(e.id);
    names.push_back(e.name);
    gray.push_back(data::load_gray(e.image));
    const bool same_width = gray.back().width == gray.front().width;
    const bool same_height = gray.back().height == gray.front().height;
    if (!same_width || !same_height) {
      throw IoError("image " + e.image.string() + ": size " + std::to_string(gray.back().width) +
                    "x" + std::to_string(gray.back().height) + " differs from the first image's " +
                    std::to_string(gray.front().width) + "x" + std::to_string(gray.front().height));
    }
    gt.push_back(data::load_camera(d, e, gray.back().width, gray.back().height));
    K.push_back(gt.back().K);
    if (d.use_masks) masks.push_back(data::load_mask(e.mask, d.mask_object_white));
    if (data::exif_orientation(e.image) != 1) ++rotated;
  }
  if (d.camera.enabled()) {
    log.info("dataset.camera", {{"fx", K.front()(0, 0)},
                                {"fy", K.front()(1, 1)},
                                {"cx", K.front()(0, 2)},
                                {"cy", K.front()(1, 2)},
                                {"exif", d.camera.exif}});
  }
  if (rotated > 0) {
    log.warn("dataset.exif_orientation",
             {{"images", rotated},
              {"note", "EXIF says these photos are displayed rotated; they are read as stored"}});
  }
  res.timings.load = seconds_since(t0);
  log.info("stage.load", {{"images", static_cast<int>(ids.size())}, {"seconds", res.timings.load}});

  const auto dump = [&](Stage s) { return cfg.output.dump_stages || s < opt.to; };

  // --- features ------------------------------------------------------------------------------
  std::vector<features::Features> feats;
  if (opt.from <= Stage::kFeatures) {
    begin(Stage::kFeatures);
    t0 = Clock::now();
    if (is_cuda(cfg.features.method)) {
      // The CUDA methods reuse one detector and its device buffers: one image after another.
      for (std::size_t i = 0; i < gray.size(); ++i) {
        check_cancel();
        feats.push_back(feature_method(gray[i], d.use_masks ? &masks[i] : nullptr, cfg.features));
      }
    } else {
      // The CPU methods keep no state: images in parallel, each result in its own slot (the same
      // output as one after another). Bunny SIFT on 16 threads: 21 s -> see findings.
      check_cancel();
      feats.resize(gray.size());
      core::parallel_for(gray.size(), [&](std::size_t i) {
        feats[i] = feature_method(gray[i], d.use_masks ? &masks[i] : nullptr, cfg.features);
      });
      check_cancel();
    }
    res.timings.features = seconds_since(t0);
    end(Stage::kFeatures, res.timings.features);
    if (dump(Stage::kFeatures)) save_features(stages / "features", feats);
    log.info("stage.features",
             {{"method", cfg.features.method}, {"seconds", res.timings.features}});
  } else {
    feats = load_features(stages / "features", ids.size());
  }
  // Lens distortion: the stages below work on undistorted keypoints (a pinhole camera); feats_raw
  // keeps the recorded positions for the colours, the export and the distortion search.
  recon::Radial dist{d.k1, d.k2};
  std::vector<features::Features> feats_raw;
  if (dist.any() || cfg.ba.refine_distortion) feats_raw = feats;
  if (dist.any()) feats = recon::undistort_features(feats_raw, K, dist);
  if (opt.to == Stage::kFeatures) {
    res.timings.total = seconds_since(t_total);
    write_text(res.run_dir / "timings.json", to_json(res.timings));
    return res;
  }

  // --- matching ------------------------------------------------------------------------------
  std::vector<matching::PairMatches> pairs;
  if (opt.from <= Stage::kMatching) {
    begin(Stage::kMatching);
    t0 = Clock::now();
    pairs = match_method(feats, cfg.matching, K, cfg.run.seed);
    res.timings.matching = seconds_since(t0);
    end(Stage::kMatching, res.timings.matching);
    if (dump(Stage::kMatching)) save_matches(stages / "matching", pairs);
    log.info("stage.matching", {{"method", cfg.matching.method},
                                {"pairs", static_cast<int>(pairs.size())},
                                {"seconds", res.timings.matching}});
  } else {
    pairs = load_matches(stages / "matching");
  }

  // --- tracks --------------------------------------------------------------------------------
  matching::Tracks tracks;
  if (opt.from <= Stage::kTracks && opt.to >= Stage::kTracks) {
    begin(Stage::kTracks);
    t0 = Clock::now();
    std::vector<std::size_t> counts;
    counts.reserve(feats.size());
    for (const auto& f : feats) counts.push_back(f.size());
    tracks = matching::build_tracks(pairs, counts);
    res.timings.tracks = seconds_since(t0);
    end(Stage::kTracks, res.timings.tracks);
    if (dump(Stage::kTracks)) save_tracks(stages / "tracks", tracks);
    log.info("stage.tracks", {{"tracks", tracks.count},
                              {"observations", static_cast<int>(tracks.size())},
                              {"seconds", res.timings.tracks}});
  } else if (opt.to >= Stage::kTracks) {
    tracks = load_tracks(stages / "tracks");
  }
  if (opt.to <= Stage::kTracks) {
    res.timings.total = seconds_since(t_total);
    write_text(res.run_dir / "timings.json", to_json(res.timings));
    return res;
  }

  // --- reconstruction ------------------------------------------------------------------------
  const recon::TrackIndex index(tracks, feats);
  // Recorded keypoint positions (differ from index only with lens distortion): for pixel lookups.
  std::optional<recon::TrackIndex> raw_store;
  if (dist.any()) raw_store.emplace(tracks, feats_raw);
  const recon::TrackIndex& raw_index = raw_store ? *raw_store : index;
  std::optional<recon::TrackIndex> refined_store;  // set by refine_distortion
  recon::Reconstruction rec;
  recon::IncrementalReport rep;
  // Gray values for snapshots and export (cheap: one pixel per track).
  const std::vector<std::uint8_t> point_gray = track_gray(raw_index, gray);
  if (opt.from <= Stage::kReconstruction) {
    begin(Stage::kReconstruction);
    t0 = Clock::now();
    recon::IncrementalParams ip;
    const auto& rc = cfg.reconstruction;
    ip.init_min_inliers = static_cast<int>(rc.init_min_inliers);
    ip.init_min_angle_deg = rc.init_min_angle_deg;
    ip.pnp.threshold_px = rc.pnp.threshold_px;
    ip.pnp.min_inliers = static_cast<int>(rc.pnp.min_inliers);
    ip.pnp.max_iters = static_cast<int>(rc.pnp.max_iters);
    ip.pnp.min_inlier_ratio = rc.pnp.min_inlier_ratio;
    ip.min_correspondences = static_cast<int>(rc.min_correspondences);
    ip.tri_min_angle_deg = rc.triangulation.min_angle_deg;
    ip.tri_max_reproj_px = rc.triangulation.max_reproj_px;
    ip.ba_every_n_cameras = static_cast<int>(cfg.ba.every_n_cameras);
    ip.ba_local_cameras = static_cast<int>(cfg.ba.local_cameras);
    ip.ba_global_growth = cfg.ba.global_growth;
    ip.seed = cfg.run.seed;
    if (rc.rig) {
      ip.rig = std::make_shared<const data::Rig>(data::load_rig(cfg.dataset.root / "rig.txt", ids));
      log.info("recon.rig",
               {{"sensors", static_cast<int>(ip.rig->sensors())}, {"frames", ip.rig->frames}});
    }
    ba::Report last_ba;
    const recon::BundleAdjuster ba = [&](recon::Reconstruction& r,
                                         const std::vector<int>& free_images) {
      const auto tb = Clock::now();
      const ba::Report br = ba_method(r, index, cfg.ba, free_images);
      const double sec = seconds_since(tb);
      res.timings.ba += sec;
      last_ba = br;
      log.debug("ba.solve", {{"seconds", sec},
                             {"cameras", r.registered_count()},
                             {"free", free_images.empty() ? r.registered_count()
                                                          : static_cast<int>(free_images.size())},
                             {"iterations", br.iterations},
                             {"initial_cost", br.initial_cost},
                             {"final_cost", br.final_cost}});
    };
    int sequence = 0;
    const recon::ReconObserver observer = [&](const recon::Reconstruction& r, recon::ReconEvent e) {
      check_cancel();
      if (sink == nullptr) return;
      const SceneSnapshot snap = make_snapshot(r, index, ids, point_gray, sequence++);
      if (e == recon::ReconEvent::kBundleAdjusted) {
        sink->ba_done(snap, last_ba);
      } else {
        sink->camera_registered(snap);
      }
    };
    rec = recon::reconstruct_incremental(pairs, feats, index, K, ip, ba, &log, &rep,
                                         sink != nullptr ? observer : recon::ReconObserver{});
    if (cfg.ba.final_max_iters > 0 && rec.registered_count() >= 2) {
      // Final refinement with a tighter robust loss (courtyard: 0.050 -> 0.035 deg median).
      config::BaConfig fc = cfg.ba;
      fc.huber_px = cfg.ba.final_huber_px;
      fc.max_iters = cfg.ba.final_max_iters;
      const auto tb = Clock::now();
      last_ba = ba_method(rec, index, fc, {});
      res.timings.ba += seconds_since(tb);
      ++rep.ba_runs;  // counted with the loop's BAs (metrics.json ba_runs)
      log.info("recon.final_ba", {{"huber_px", fc.huber_px},
                                  {"iterations", last_ba.iterations},
                                  {"initial_cost", last_ba.initial_cost},
                                  {"final_cost", last_ba.final_cost}});
      if (cfg.ba.refine_focal) {
        const auto tf = Clock::now();
        const FocalSearch fs_res = refine_focal(rec, index, fc, ba_method);
        res.timings.ba += seconds_since(tf);
        log.info("recon.focal_refined", {{"scale", fs_res.scale},
                                         {"fx", rec.cameras.front().K(0, 0)},
                                         {"cost_before", last_ba.final_cost},
                                         {"cost_after", fs_res.cost},
                                         {"evaluations", fs_res.evaluations}});
      }
      const double coverage = cfg.ba.refine_distortion ? keypoint_coverage(feats_raw, K) : 0.0;
      if (cfg.ba.refine_distortion && coverage < kMinDistortionCoverage) {
        // Keypoints near the image centre only (e.g. a masked object): radial distortion barely
        // moves them and k1 trades off against the focal length. Measured on the masked bunny:
        // k1 = -0.15 estimated as -0.345.
        log.warn("recon.distortion_not_refined",
                 {{"r2_p90", coverage},
                  {"needs", kMinDistortionCoverage},
                  {"note", "keypoints cover only the image centre; k1 left as configured"}});
      } else if (cfg.ba.refine_distortion) {
        const auto td = Clock::now();
        DistortionSearch ds =
            refine_distortion(rec, tracks, feats_raw, dist.k1, dist.k2, fc, ba_method);
        res.timings.ba += seconds_since(td);
        log.info("recon.distortion_refined", {{"k1", ds.k1},
                                              {"k2", dist.k2},
                                              {"cost_recorded_px", ds.cost},
                                              {"cost_at_configured_k1", ds.cost_start},
                                              {"evaluations", ds.evaluations}});
        dist.k1 = ds.k1;
        feats = std::move(ds.feats);
        refined_store.emplace(tracks, feats);
        if (!raw_store) raw_store.emplace(tracks, feats_raw);
      }
      if (sink != nullptr) {
        sink->ba_done(make_snapshot(rec, index, ids, point_gray, sequence++), last_ba);
      }
    }
    res.timings.reconstruction = seconds_since(t0);
    end(Stage::kReconstruction, res.timings.reconstruction);
    if (dump(Stage::kReconstruction)) {
      save_reconstruction(stages / "reconstruction", rec);
      // The loop's counters, so a later --from export reproduces metrics.json exactly.
      core::npy_write(stages / "reconstruction" / "report.npy",
                      core::NpyArray::from<std::int32_t>(
                          {3}, {rep.pnp_attempts, rep.pnp_failures, rep.ba_runs}));
      // The distortion the reconstruction was made with (refine_distortion may change k1).
      core::npy_write(stages / "reconstruction" / "distortion.npy",
                      core::NpyArray::from<double>({2}, {dist.k1, dist.k2}));
    }
  } else {
    rec = load_reconstruction(stages / "reconstruction", K);
    const auto r =
        core::npy_read(stages / "reconstruction" / "report.npy").to_vector<std::int32_t>();
    if (r.size() != 3) throw IoError("stages/reconstruction/report.npy: expected 3 values");
    rep.pnp_attempts = r[0];
    rep.pnp_failures = r[1];
    rep.ba_runs = r[2];
    const fs::path dfile = stages / "reconstruction" / "distortion.npy";
    if (fs::exists(dfile)) {
      const auto k = core::npy_read(dfile).to_vector<double>();
      if (k.size() == 2 && (k[0] != dist.k1 || k[1] != dist.k2)) {
        dist = {k[0], k[1]};
        if (feats_raw.empty()) feats_raw = feats;
        feats = recon::undistort_features(feats_raw, K, dist);
        refined_store.emplace(tracks, feats);
        if (!raw_store) raw_store.emplace(tracks, feats_raw);
      }
    }
  }
  const recon::TrackIndex& final_index = refined_store ? *refined_store : index;
  const recon::TrackIndex& final_raw_index = raw_store ? *raw_store : index;
  const std::size_t n_pairs = pairs.size();
  res.metrics = compute_metrics(rec, final_index, feats, n_pairs, tracks.count, ids, gt, rep);
  {
    RunModel m;
    m.names = names;
    m.k1 = dist.k1;
    m.k2 = dist.k2;
    for (const auto& c : rec.cameras) {
      m.registered.push_back(c.registered);
      m.K.push_back(c.K);
      m.R.push_back(c.pose.R);
      m.t.push_back(c.pose.t);
    }
    for (std::size_t t = 0; t < rec.points.size(); ++t) {
      if (rec.has_point[t] != 0) m.points.push_back(rec.points[t]);
    }
    res.model = std::move(m);
  }
  write_text(res.run_dir / "metrics.json", to_json(*res.metrics));
  log.info("stage.reconstruction",
           {{"registered", static_cast<int>(res.metrics->registered_ids.size())},
            {"points", res.metrics->points},
            {"reproj_median_px", res.metrics->reproj_median_px},
            {"rotation_median_deg", res.metrics->rotation_median_deg},
            {"seconds", res.timings.reconstruction}});

  // --- export --------------------------------------------------------------------------------
  if (opt.to >= Stage::kExport) {
    begin(Stage::kExport);
    t0 = Clock::now();
    std::vector<fs::path> sources;
    sources.reserve(files.size());
    for (const auto& e : files) sources.push_back(e.image);
    // Colours from the images (the stages before work on gray); only when something is written.
    const bool any_export = cfg.output.ply || cfg.output.colmap || cfg.output.colmap_scene;
    const std::vector<exporter::Rgb> colours =
        any_export ? track_rgb(final_raw_index, sources) : std::vector<exporter::Rgb>{};
    const int w = gray.front().width;
    const int h = gray.front().height;
    if (res.model && !colours.empty()) {
      for (std::size_t t = 0; t < rec.points.size(); ++t) {
        if (rec.has_point[t] != 0) res.model->colours.push_back(colours[t]);
      }
    }
    if (cfg.output.ply) exporter::write_ply(res.run_dir / "model.ply", rec, colours);
    if (cfg.output.colmap) {
      // Recorded keypoints with the OPENCV model when there is distortion, else pinhole.
      exporter::write_colmap(res.run_dir / "colmap", rec, final_index,
                             dist.any() ? feats_raw : feats, names, w, h, colours, dist);
    }
    if (cfg.output.colmap_scene) {
      exporter::write_colmap_scene(res.run_dir / "scene", rec, final_index, feats, sources, w, h,
                                   colours, dist);
    }
    res.timings.export_ = seconds_since(t0);
    end(Stage::kExport, res.timings.export_);
  }
  res.timings.total = seconds_since(t_total);
  write_text(res.run_dir / "timings.json", to_json(res.timings));
  log.info("run.done", {{"seconds", res.timings.total}});
  if (sink != nullptr) sink->finished(*res.metrics, res.timings);
  return res;
}

}  // namespace

// With ba.refine_distortion the reconstruction is made with the configured k1, so a strong
// distortion can already have cost registrations (ETH3D courtyard, k1 = -0.10: 23/38). Each pass
// re-runs with the estimate until it moves by less than 0.005 (at most 3 passes).
RunResult run_pipeline(const config::Config& cfg, const RunOptions& opt, core::Logger& log,
                       const Registry& registry, ProgressSink* sink) {
  if (!cfg.ba.refine_distortion || opt.from > Stage::kFeatures) {
    return run_once(cfg, opt, log, registry, sink);
  }
  config::Config c = cfg;
  RunResult r;
  for (int pass = 1;; ++pass) {
    r = run_once(c, opt, log, registry, sink);
    if (!r.model || pass == 3 || std::abs(r.model->k1 - c.dataset.k1) < 0.005) break;
    log.info("run.distortion_pass", {{"pass", pass + 1}, {"k1", r.model->k1}});
    c.dataset.k1 = r.model->k1;
  }
  return r;
}

std::string to_json(const Metrics& m) {
  nlohmann::ordered_json j;
  j["images"] = m.images;
  j["keypoints_mean"] = m.keypoints_mean;
  j["verified_pairs"] = m.verified_pairs;
  j["tracks"] = m.tracks;
  j["init_pair"] = {m.init_i, m.init_j};
  j["registered"] = m.registered_ids.size();
  j["registered_ids"] = m.registered_ids;
  j["points"] = m.points;
  j["observations"] = m.observations;
  j["reprojection_px"] = {
      {"median", m.reproj_median_px}, {"mean", m.reproj_mean_px}, {"p95", m.reproj_p95_px}};
  j["rotation_error_deg"] = {{"median", m.rotation_median_deg}, {"max", m.rotation_max_deg}};
  j["centre_error_pct"] = {{"median", m.centre_median_pct}, {"max", m.centre_max_pct}};
  j["pnp_failures"] = m.pnp_failures;
  j["ba_runs"] = m.ba_runs;
  return j.dump(2) + "\n";
}

std::string to_json(const Timings& t) {
  nlohmann::ordered_json j;
  j["load"] = t.load;
  j["features"] = t.features;
  j["matching"] = t.matching;
  j["tracks"] = t.tracks;
  j["reconstruction"] = t.reconstruction;
  j["ba"] = t.ba;
  j["export"] = t.export_;
  j["total"] = t.total;
  return j.dump(2) + "\n";
}

}  // namespace sfm::pipeline
