#include "sfm/recon/incremental.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>

#include "sfm/core/error.hpp"
#include "sfm/core/trace.hpp"

namespace sfm::recon {

namespace {

// Adds the time from construction to destruction to `acc` (seconds), and marks the same span as
// a named profiler range.
class Stopwatch {
 public:
  Stopwatch(double& acc, const char* name)
      : range_(name), acc_(acc), t0_(std::chrono::steady_clock::now()) {}
  Stopwatch(const Stopwatch&) = delete;
  Stopwatch& operator=(const Stopwatch&) = delete;
  Stopwatch(Stopwatch&&) = delete;
  Stopwatch& operator=(Stopwatch&&) = delete;
  ~Stopwatch() {
    acc_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  core::TraceRange range_;
  double& acc_;
  std::chrono::steady_clock::time_point t0_;
};

// Triangulation RANSAC over view pairs: all pairs up to this many (tracks of <= 23 views),
// otherwise sampled pairs, at most kMaxPairTrials.
constexpr std::size_t kExhaustivePairs = 256;
constexpr std::size_t kMaxPairTrials = 256;

struct ViewObs {
  const TrackObs* o;
  const Camera* c;
  Eigen::Vector2d xn;
};

// Observations of `views` consistent with X (in front, reprojection <= max_px).
std::vector<std::size_t> consistent(const std::vector<ViewObs>& views, const Eigen::Vector3d& X,
                                    double max_px) {
  std::vector<std::size_t> in;
  for (std::size_t k = 0; k < views.size(); ++k) {
    const Camera& c = *views[k].c;
    if (depth(c.pose, X) > 0 && reprojection_error(c.K, c.pose, X, views[k].o->uv) <= max_px) {
      in.push_back(k);
    }
  }
  return in;
}

bool dlt(const std::vector<ViewObs>& views, const std::vector<std::size_t>& use,
         Eigen::Vector3d& X) {
  std::vector<Mat34> P;
  std::vector<Eigen::Vector2d> xn;
  for (const std::size_t k : use) {
    P.push_back(pose_matrix(views[k].c->pose));
    xn.push_back(views[k].xn);
  }
  return triangulate_dlt(P, xn, X);
}

double widest_angle(const std::vector<ViewObs>& views, const std::vector<std::size_t>& use,
                    const Eigen::Vector3d& X) {
  double widest = 0.0;
  for (std::size_t a = 0; a < use.size(); ++a) {
    for (std::size_t b = a + 1; b < use.size(); ++b) {
      widest = std::max(
          widest, ray_angle_deg(views[use[a]].c->pose.centre(), views[use[b]].c->pose.centre(), X));
    }
  }
  return widest;
}

}  // namespace

bool triangulate_track(Reconstruction& rec, const TrackIndex& index, int t,
                       const IncrementalParams& p) {
  if (rec.obs_outlier.size() != index.obs_count()) rec.obs_outlier.assign(index.obs_count(), 0);
  std::vector<ViewObs> views;
  for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
    const Camera& c = rec.cameras[static_cast<std::size_t>(o->image)];
    if (c.registered) views.push_back({o, &c, normalise(c.K, o->uv)});
  }
  if (views.size() < 2) return false;
  std::vector<std::size_t> all(views.size());
  std::iota(all.begin(), all.end(), std::size_t{0});

  // All views agree: the common case.
  Eigen::Vector3d X;
  std::vector<std::size_t> inl;
  if (dlt(views, all, X)) inl = consistent(views, X, p.tri_max_reproj_px);
  if (inl.size() < views.size() && views.size() >= 3) {
    // Some observation is wrong: every pair of views proposes a point; keep the one most views
    // agree with (ties: the first pair), then refit on those views (RANSAC over pairs).
    std::vector<std::size_t> best;
    const auto try_pair = [&](std::size_t a, std::size_t b) {
      Eigen::Vector3d Y;
      if (!dlt(views, {a, b}, Y)) return;
      if (ray_angle_deg(views[a].c->pose.centre(), views[b].c->pose.centre(), Y) <
          p.tri_min_angle_deg) {
        return;
      }
      auto c = consistent(views, Y, p.tri_max_reproj_px);
      if (c.size() > best.size()) best = std::move(c);
    };
    const std::size_t n = views.size();
    if (n * (n - 1) / 2 <= kExhaustivePairs) {
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) try_pair(a, b);
      }
    } else {
      // Long track (P9c): every pair costs O(n) to score, all pairs O(n^3) — 99 % of the
      // triangulation time on the 4-camera scenes. Sample pairs instead (seeded by the track, so
      // runs repeat), stopping when a pair of inliers has been drawn with 99.9 % confidence.
      std::uint64_t state = 0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(t);
      const auto next = [&state](std::size_t m) {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return static_cast<std::size_t>((z ^ (z >> 31)) % m);
      };
      for (std::size_t trial = 0; trial < kMaxPairTrials; ++trial) {
        const std::size_t a = next(n);
        std::size_t b = next(n - 1);
        if (b >= a) ++b;
        try_pair(std::min(a, b), std::max(a, b));
        const double w = static_cast<double>(best.size()) / static_cast<double>(n);
        if (w > 0.0 && w < 1.0) {
          const double need = std::log(1.0 - 0.999) / std::log(1.0 - w * w);
          if (static_cast<double>(trial + 1) >= need) break;
        } else if (w >= 1.0) {
          break;
        }
      }
    }
    if (best.size() >= 2 && dlt(views, best, X)) inl = consistent(views, X, p.tri_max_reproj_px);
  }
  if (inl.size() < 2 || widest_angle(views, inl, X) < p.tri_min_angle_deg) return false;
  for (std::size_t k = 0; k < views.size(); ++k) rec.obs_outlier[index.index_of(views[k].o)] = 1;
  for (const std::size_t k : inl) rec.obs_outlier[index.index_of(views[k].o)] = 0;
  rec.points[static_cast<std::size_t>(t)] = X;
  rec.has_point[static_cast<std::size_t>(t)] = 1;
  return true;
}

int filter_points(Reconstruction& rec, const TrackIndex& index, double max_px) {
  if (rec.obs_outlier.size() != index.obs_count()) rec.obs_outlier.assign(index.obs_count(), 0);
  int removed = 0;
  for (int t = 0; t < index.track_count(); ++t) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    const Eigen::Vector3d& X = rec.points[static_cast<std::size_t>(t)];
    int good = 0;
    for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
      if (!observation_used(rec, index, o)) continue;
      const Camera& c = rec.cameras[static_cast<std::size_t>(o->image)];
      if (depth(c.pose, X) <= 0 || reprojection_error(c.K, c.pose, X, o->uv) > max_px) {
        rec.obs_outlier[index.index_of(o)] = 1;  // this observation only
      } else {
        ++good;
      }
    }
    if (good < 2) {  // the point itself is unsupported: drop it, reconsider all its observations
      rec.has_point[static_cast<std::size_t>(t)] = 0;
      for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
        rec.obs_outlier[index.index_of(o)] = 0;
      }
      ++removed;
    }
  }
  return removed;
}

void check_new_observations(Reconstruction& rec, const TrackIndex& index, int image,
                            double max_px) {
  if (rec.obs_outlier.size() != index.obs_count()) rec.obs_outlier.assign(index.obs_count(), 0);
  const Camera& c = rec.cameras[static_cast<std::size_t>(image)];
  for (const auto& [t, kp] : index.in_image(image)) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
      if (o->image != image) continue;
      const Eigen::Vector3d& X = rec.points[static_cast<std::size_t>(t)];
      const bool bad = depth(c.pose, X) <= 0 || reprojection_error(c.K, c.pose, X, o->uv) > max_px;
      rec.obs_outlier[index.index_of(o)] = bad ? 1 : 0;
    }
  }
}

namespace {

// 2D-3D correspondences of image i: tracked keypoints whose track has a point.
void correspondences(const Reconstruction& rec, const TrackIndex& index, int i,
                     std::vector<Eigen::Vector3d>* X, std::vector<Eigen::Vector2d>* uv,
                     int* count) {
  int n = 0;
  for (const auto& [t, kp] : index.in_image(i)) {
    if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
    ++n;
    if (X == nullptr) continue;
    X->push_back(rec.points[static_cast<std::size_t>(t)]);
    for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
      if (o->image == i) {
        uv->push_back(o->uv);
        break;
      }
    }
  }
  *count = n;
}

std::uint64_t mix(std::uint64_t seed, std::uint64_t a, std::uint64_t b) {
  std::uint64_t z = seed ^ (a << 32) ^ b;
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

double rig_scale(const Reconstruction& rec, const TrackIndex& index,
                 const std::vector<int>& images) {
  const data::Rig& rig = *rec.rig;
  struct Obs {
    int sibling;
    int frame_of;  // the registered image whose frame pose places the sibling
    Eigen::Vector3d X;
    Eigen::Vector2d uv;
  };
  std::vector<Obs> obs;
  for (const int a : images) {
    const int f = rig.frame[static_cast<std::size_t>(a)];
    for (int b = 0; b < static_cast<int>(rig.frame.size()); ++b) {
      if (b == a || rig.frame[static_cast<std::size_t>(b)] != f ||
          rec.cameras[static_cast<std::size_t>(b)].registered) {
        continue;
      }
      for (const auto& [t, kp] : index.in_image(b)) {
        if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
        for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
          if (o->image == b) obs.push_back({b, a, rec.points[static_cast<std::size_t>(t)], o->uv});
        }
      }
    }
  }
  if (obs.size() < 20) return 0.0;
  const auto cost = [&](double log_u) {
    const double u = std::pow(10.0, log_u);
    data::Rig scaled = rig;
    for (auto& t : scaled.t) t *= u;
    double c = 0.0;
    for (const Obs& o : obs) {
      const Pose fp = rig_frame_pose(scaled, o.frame_of,
                                     rec.cameras[static_cast<std::size_t>(o.frame_of)].pose);
      const Pose sp = rig_image_pose(scaled, o.sibling, fp);
      const Camera& cam = rec.cameras[static_cast<std::size_t>(o.sibling)];
      const double e = depth(sp, o.X) > 0 ? reprojection_error(cam.K, sp, o.X, o.uv) : 1e9;
      const double r = std::min(e, 10.0);
      c += r * r;
    }
    return c;
  };
  double best = -4.0;
  double best_c = cost(best);
  for (double l = -4.0; l <= 4.0 + 1e-9; l += 0.02) {
    const double c = cost(l);
    if (c < best_c) {
      best_c = c;
      best = l;
    }
  }
  double lo = best - 0.02;
  double hi = best + 0.02;
  for (int it = 0; it < 60; ++it) {  // golden section
    const double g = 0.381966011250105;
    const double m1 = lo + g * (hi - lo);
    const double m2 = hi - g * (hi - lo);
    if (cost(m1) < cost(m2)) {
      hi = m2;
    } else {
      lo = m1;
    }
  }
  return std::pow(10.0, 0.5 * (lo + hi));
}

double rig_scale_stereo(const Reconstruction& rec, const TrackIndex& index,
                        const std::vector<int>& images, double min_angle_deg) {
  const data::Rig& rig = *rec.rig;
  std::vector<double> ratio;
  for (const int a : images) {
    const auto ua = static_cast<std::size_t>(a);
    const Camera& ca = rec.cameras[ua];
    const auto sa = static_cast<std::size_t>(rig.sensor[ua]);
    for (int b = 0; b < static_cast<int>(rig.frame.size()); ++b) {
      const auto ub = static_cast<std::size_t>(b);
      if (b == a || rig.frame[ub] != rig.frame[ua]) continue;
      const auto sb = static_cast<std::size_t>(rig.sensor[ub]);
      // Metric pose of b in a's camera frame: x_b = R_ab x_a + t_ab.
      const Eigen::Matrix3d R_ab = rig.R[sb] * rig.R[sa].transpose();
      const Eigen::Vector3d t_ab = rig.t[sb] - R_ab * rig.t[sa];
      const Pose pa{Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
      const Pose pb{R_ab, t_ab};
      const std::vector<Mat34> P = {pose_matrix(pa), pose_matrix(pb)};
      const Camera& cb = rec.cameras[ub];
      for (const auto& [t, kp] : index.in_image(a)) {
        if (rec.has_point[static_cast<std::size_t>(t)] == 0) continue;
        const TrackObs* oa = nullptr;
        const TrackObs* ob = nullptr;
        for (const TrackObs* o = index.begin(t); o != index.end(t); ++o) {
          if (o->image == a) oa = o;
          if (o->image == b) ob = o;
        }
        if (oa == nullptr || ob == nullptr) continue;
        Eigen::Vector3d Y;
        if (!triangulate_dlt(P, {normalise(ca.K, oa->uv), normalise(cb.K, ob->uv)}, Y)) continue;
        if (Y.z() <= 0 || depth(pb, Y) <= 0) continue;
        if (ray_angle_deg(pa.centre(), pb.centre(), Y) < min_angle_deg) continue;
        const double z_model = depth(ca.pose, rec.points[static_cast<std::size_t>(t)]);
        if (z_model <= 0) continue;
        ratio.push_back(z_model / Y.z());
      }
    }
  }
  if (ratio.size() < 20) return 0.0;
  const auto mid = ratio.begin() + static_cast<std::ptrdiff_t>(ratio.size() / 2);
  std::nth_element(ratio.begin(), mid, ratio.end());
  return *mid;
}

std::vector<int> register_rig_siblings(Reconstruction& rec, const TrackIndex& index, int image,
                                       const IncrementalParams& p) {
  std::vector<int> out;
  if (!rec.rig) return out;
  const data::Rig& rig = *rec.rig;
  const int f = rig.frame[static_cast<std::size_t>(image)];
  const Pose fp = rig_frame_pose(rig, image, rec.cameras[static_cast<std::size_t>(image)].pose);
  for (int b = 0; b < static_cast<int>(rig.frame.size()); ++b) {
    Camera& cam = rec.cameras[static_cast<std::size_t>(b)];
    if (rig.frame[static_cast<std::size_t>(b)] != f || cam.registered) continue;
    cam.pose = rig_image_pose(rig, b, fp);
    cam.registered = true;
    rec.order.push_back(b);
    out.push_back(b);
  }
  for (const int b : out) check_new_observations(rec, index, b, p.tri_max_reproj_px);
  for (const int b : out) {
    for (const auto& [t, kp] : index.in_image(b)) {
      if (rec.has_point[static_cast<std::size_t>(t)] == 0) triangulate_track(rec, index, t, p);
    }
  }
  return out;
}

Reconstruction reconstruct_incremental(const std::vector<matching::PairMatches>& pairs,
                                       const std::vector<features::Features>& features,
                                       const TrackIndex& index,
                                       const std::vector<Eigen::Matrix3d>& K,
                                       const IncrementalParams& p, const BundleAdjuster& ba,
                                       core::Logger* log, IncrementalReport* report,
                                       const ReconObserver& observer) {
  const auto notify = [&](const Reconstruction& r, ReconEvent e) {
    if (observer) observer(r, e);
  };
  IncrementalReport rep;
  const int n_images = static_cast<int>(features.size());
  Reconstruction rec;
  rec.cameras.resize(features.size());
  for (std::size_t i = 0; i < features.size(); ++i) rec.cameras[i].K = K[i];
  rec.rig = p.rig;
  if (rec.rig && rec.rig->frame.size() != features.size()) {
    throw GeometryError("reconstruct_incremental: the rig has " +
                        std::to_string(rec.rig->frame.size()) + " images, the dataset " +
                        std::to_string(features.size()));
  }
  rec.points.assign(static_cast<std::size_t>(index.track_count()), Eigen::Vector3d::Zero());
  rec.has_point.assign(static_cast<std::size_t>(index.track_count()), 0);

  bool have_pair = false;
  {
    const Stopwatch sw(rep.seconds.init_pair, "recon.init_pair");
    have_pair = choose_initial_pair(pairs, features, K, p, rep.init);
  }
  if (!have_pair) {
    throw GeometryError(
        "reconstruct_incremental: no initial pair with >= " + std::to_string(p.init_min_inliers) +
        " points and median angle >= " + std::to_string(p.init_min_angle_deg) + " deg");
  }
  const InitialPair& ip = rep.init;
  rec.cameras[static_cast<std::size_t>(ip.i)].registered = true;
  rec.cameras[static_cast<std::size_t>(ip.j)].pose = ip.pose_j;
  rec.cameras[static_cast<std::size_t>(ip.j)].registered = true;
  rec.order = {ip.i, ip.j};
  int init_points = 0;
  {
    const Stopwatch sw(rep.seconds.init_points, "recon.init_points");
    for (int t = 0; t < index.track_count(); ++t) {
      init_points += triangulate_track(rec, index, t, p) ? 1 : 0;
    }
  }
  if (log != nullptr) {
    log->info("recon.init", {{"i", ip.i},
                             {"j", ip.j},
                             {"pair_points", ip.points},
                             {"median_angle_deg", ip.median_angle_deg},
                             {"points", init_points}});
  }

  if (rec.rig) {
    // The pair fixes the model's scale arbitrarily (|t_j| = 1); the rig's offsets are metric.
    // Rescale the model so that both agree, then pose the pair's siblings from the rig.
    const double u_reproj = rig_scale(rec, index, {ip.i, ip.j});
    const double u_stereo = rig_scale_stereo(rec, index, {ip.i, ip.j});
    const double u = u_stereo > 0.0 ? u_stereo : u_reproj;
    rep.rig_scale = u;
    if (u > 0.0) {
      for (auto& X : rec.points) X /= u;
      for (Camera& c : rec.cameras) {
        if (c.registered) c.pose.t /= u;
      }
    }
    int sib = 0;
    for (const int a : {ip.i, ip.j}) {
      sib += static_cast<int>(register_rig_siblings(rec, index, a, p).size());
    }
    rep.rig_registered += sib;
    if (log != nullptr) {
      log->info("recon.rig_init", {{"scale", u},
                                   {"scale_stereo", u_stereo},
                                   {"scale_reprojection", u_reproj},
                                   {"siblings", sib},
                                   {"points", rec.point_count()}});
    }
    if (u <= 0.0) {
      throw GeometryError(
          "reconstruct_incremental: the initial pair's rig siblings see fewer "
          "than 20 points; cannot find the rig scale");
    }
  }
  notify(rec, ReconEvent::kInitialized);

  int global_at = 0;  // registered cameras at the last global BA
  const auto run_ba = [&](bool global) {
    if (!ba) return;
    std::vector<int> free_images;
    const int n_reg = rec.registered_count();
    if (!global) {
      const std::size_t n =
          std::min(rec.order.size(), static_cast<std::size_t>(p.ba_local_cameras));
      free_images.assign(rec.order.end() - static_cast<std::ptrdiff_t>(n), rec.order.end());
    } else {
      global_at = n_reg;
      ++rep.ba_global;
    }
    {
      const Stopwatch sw(rep.seconds.ba, "recon.ba");
      ba(rec, free_images);
    }
    ++rep.ba_runs;
    {
      const Stopwatch sw(rep.seconds.filter, "recon.filter");
      rep.points_filtered += filter_points(rec, index, p.tri_max_reproj_px);
    }
    int added = 0;
    {
      const Stopwatch sw(rep.seconds.retriangulate, "recon.retriangulate");
      for (int t = 0; t < index.track_count(); ++t) {
        if (rec.has_point[static_cast<std::size_t>(t)] == 0) {
          added += triangulate_track(rec, index, t, p) ? 1 : 0;
        }
      }
    }
    if (log != nullptr) {
      log->info("recon.ba",
                {{"cameras", rec.registered_count()},
                 {"free", global ? rec.registered_count() : static_cast<int>(free_images.size())},
                 {"points", rec.point_count()},
                 {"retriangulated", added}});
    }
    notify(rec, ReconEvent::kBundleAdjusted);
  };

  std::set<int> failed;  // PnP failed since the last BA
  int since_ba = 1;      // the initial pair has not been adjusted yet
  int since_global = 1;  // registrations since the last global BA
  if (ba) {
    run_ba(true);
    since_ba = 0;
    since_global = 0;
  }
  for (;;) {
    int best = -1;
    int best_count = 0;
    {
      const Stopwatch sw(rep.seconds.select, "recon.select");
      for (int i = 0; i < n_images; ++i) {
        if (rec.cameras[static_cast<std::size_t>(i)].registered || failed.count(i) != 0) continue;
        int c = 0;
        correspondences(rec, index, i, nullptr, nullptr, &c);
        if (c >= p.min_correspondences && c > best_count) {
          best = i;
          best_count = c;
        }
      }
    }
    if (best < 0) {
      if (since_global == 0 || !ba) break;
      run_ba(true);
      since_ba = 0;
      since_global = 0;
      failed.clear();
      continue;
    }

    std::vector<Eigen::Vector3d> X;
    std::vector<Eigen::Vector2d> uv;
    int count = 0;
    correspondences(rec, index, best, &X, &uv, &count);
    PnpParams pp = p.pnp;
    pp.seed =
        mix(p.seed, static_cast<std::uint64_t>(best), static_cast<std::uint64_t>(rep.pnp_attempts));
    ++rep.pnp_attempts;
    PnpResult r;
    {
      const Stopwatch sw(rep.seconds.pnp, "recon.pnp");
      r = pnp_ransac(K[static_cast<std::size_t>(best)], X, uv, pp);
    }
    if (!r.ok) {
      ++rep.pnp_failures;
      failed.insert(best);
      if (log != nullptr) {
        log->info("recon.pnp_failed",
                  {{"image", best}, {"correspondences", count}, {"inliers", r.inliers}});
      }
      continue;
    }
    Camera& cam = rec.cameras[static_cast<std::size_t>(best)];
    cam.pose = r.pose;
    cam.registered = true;
    rec.order.push_back(best);
    int added = 0;
    int siblings = 0;
    {
      const Stopwatch sw(rep.seconds.triangulate, "recon.triangulate");
      check_new_observations(rec, index, best, p.tri_max_reproj_px);
      for (const auto& [t, kp] : index.in_image(best)) {
        if (rec.has_point[static_cast<std::size_t>(t)] == 0) {
          added += triangulate_track(rec, index, t, p) ? 1 : 0;
        }
      }
      siblings = static_cast<int>(register_rig_siblings(rec, index, best, p).size());
    }
    rep.rig_registered += siblings;
    if (log != nullptr) {
      log->info("recon.registered", {{"image", best},
                                     {"correspondences", count},
                                     {"inliers", r.inliers},
                                     {"new_points", added},
                                     {"rig_siblings", siblings},
                                     {"cameras", rec.registered_count()}});
    }
    notify(rec, ReconEvent::kRegistered);
    since_ba += 1 + siblings;
    since_global += 1 + siblings;
    if (p.ba_every_n_cameras > 0 && since_ba >= p.ba_every_n_cameras) {
      const int n_reg = rec.registered_count();
      const bool global =
          p.ba_local_cameras <= 0 || n_reg <= p.ba_local_cameras ||
          static_cast<double>(n_reg) >= (1.0 + p.ba_global_growth) * static_cast<double>(global_at);
      run_ba(global);
      since_ba = 0;
      if (global) since_global = 0;
      failed.clear();
    }
  }
  if (log != nullptr) {
    const auto& sec = rep.seconds;
    log->info("recon.timing", {{"init_pair_s", sec.init_pair},
                               {"init_points_s", sec.init_points},
                               {"select_s", sec.select},
                               {"pnp_s", sec.pnp},
                               {"triangulate_s", sec.triangulate},
                               {"ba_s", sec.ba},
                               {"filter_s", sec.filter},
                               {"retriangulate_s", sec.retriangulate}});
  }
  if (report != nullptr) *report = rep;
  return rec;
}

}  // namespace sfm::recon
