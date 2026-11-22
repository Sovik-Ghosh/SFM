#pragma once
// Incremental reconstruction: initial pair -> repeat (next view, PnP, triangulate new tracks),
// with bundle adjustment through a callback (the BA stage lives in its own target).

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "sfm/core/logger.hpp"
#include "sfm/features/features.hpp"
#include "sfm/matching/matches.hpp"
#include "sfm/recon/pnp.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::recon {

struct IncrementalParams {
  int init_min_inliers = 50;
  double init_min_angle_deg = 5.0;
  PnpParams pnp;
  int min_correspondences = 30;
  double tri_min_angle_deg = 2.0;
  double tri_max_reproj_px = 4.0;
  int ba_every_n_cameras = 5;  // 0: BA only after the initial pair and when no camera is left
  // Local BA (> 0): the periodic BA adjusts only the ba_local_cameras most recently registered
  // cameras (and the points they see), unless the model has grown by ba_global_growth (fraction of
  // registered cameras) since the last global BA. 0: every BA is global (the P7 behaviour).
  int ba_local_cameras = 0;
  double ba_global_growth = 0.25;
  std::uint64_t seed = 0;
  // Camera rig (P9b; null = none). With a rig: after the initial pair the model is scaled to the
  // rig's metric offsets (1D search on the reprojection of the pair's siblings), and every
  // registration (initial pair, PnP) also registers the other images of that frame from the rig.
  std::shared_ptr<const data::Rig> rig;
};

// Scale u (model units per rig unit) that best explains the observations of `images`' rig
// siblings: each sibling posed as rig offset (translation x u) * frame pose of its image, robust
// cost sum min(reprojection, 10 px)^2 over its tracked points, u searched on a log grid in
// [1e-4, 1e4] and refined. Returns 0 if the siblings see fewer than 20 points.
double rig_scale(const Reconstruction& rec, const TrackIndex& index,
                 const std::vector<int>& images);

// Scale u from rig stereo (the default): every point of the model seen by one of `images` (a) and
// by a sibling b of the same frame is triangulated from a and b with the rig's metric relative
// pose; u = median(depth in the model / metric depth) over points whose two rays meet at
// >= min_angle_deg. Returns 0 with fewer than 20 such points. Unlike rig_scale it does not depend
// on the reprojection cost, which a weak initial pair (near pure rotation) can bias (playground,
// P9b: rig_scale was 2.65x off, stereo within a few %).
double rig_scale_stereo(const Reconstruction& rec, const TrackIndex& index,
                        const std::vector<int>& images, double min_angle_deg = 0.05);

// Registers the unregistered images of `image`'s rig frame from the rig (no PnP), checks their
// observations of existing points and triangulates their tracks. Returns the images registered.
std::vector<int> register_rig_siblings(Reconstruction& rec, const TrackIndex& index, int image,
                                       const IncrementalParams& p);

struct InitialPair {
  int i = -1;
  int j = -1;
  Pose pose_j;  // camera i = [I | 0], ||t_j|| = 1
  int points = 0;
  double median_angle_deg = 0.0;
};

// Pairs with >= init_min_inliers F inliers, most inliers first (ties: lower (i, j)). For each:
// E = K_j^T F K_i (nearest essential), the decomposition with most points in front, triangulate
// the inliers; accept the first pair with >= init_min_inliers points passing the triangulation
// checks (in front, reprojection <= tri_max_reproj_px) and a median ray angle >=
// init_min_angle_deg. Returns false if no pair qualifies.
bool choose_initial_pair(const std::vector<matching::PairMatches>& pairs,
                         const std::vector<features::Features>& features,
                         const std::vector<Eigen::Matrix3d>& K, const IncrementalParams& p,
                         InitialPair& out);

// Triangulates track t from its observations in registered cameras. All views first (n-view
// DLT); if some view disagrees (behind the camera or reprojection > tri_max_reproj_px), RANSAC
// over view pairs keeps the point most views agree with and refits on them. Accepted with >= 2
// agreeing views and a widest ray angle >= tri_min_angle_deg; the other observations are marked
// in rec.obs_outlier. Returns true if a point was set.
bool triangulate_track(Reconstruction& rec, const TrackIndex& index, int t,
                       const IncrementalParams& p);

// Marks observations that are behind their camera or reproject worse than max_px as outliers;
// removes points left with fewer than 2 used observations (their observations are reset, so a
// later triangulation reconsiders them). Returns the number of points removed.
int filter_points(Reconstruction& rec, const TrackIndex& index, double max_px);

// After registering `image`: its observations of existing points are used only if they agree
// with the point (in front, reprojection <= max_px).
void check_new_observations(Reconstruction& rec, const TrackIndex& index, int image, double max_px);

// free_images empty: global BA; otherwise a local BA of those images (ba::bundle_adjust).
using BundleAdjuster = std::function<void(Reconstruction&, const std::vector<int>& free_images)>;

// Progress of the loop, for observers (e.g. a live viewer). Called on the caller's thread with the
// current state; an observer may throw (e.g. sfm::Cancelled) to stop the loop.
enum class ReconEvent : std::uint8_t { kInitialized, kRegistered, kBundleAdjusted };
using ReconObserver = std::function<void(const Reconstruction&, ReconEvent)>;

struct IncrementalReport {
  InitialPair init;
  int pnp_attempts = 0;
  int pnp_failures = 0;
  int ba_runs = 0;    // all BAs
  int ba_global = 0;  // of which global
  int points_filtered = 0;
  int rig_registered = 0;  // images registered from the rig (not by PnP)
  double rig_scale = 0.0;  // the scale found after the initial pair (0: no rig)
  // Wall time (s) by part of the loop (P9c): initial pair search, first triangulation, choosing
  // the next image, PnP, triangulation after a registration (incl. rig siblings), BA (the
  // adjuster only), filtering and re-triangulation after each BA.
  struct Seconds {
    double init_pair = 0.0;
    double init_points = 0.0;
    double select = 0.0;
    double pnp = 0.0;
    double triangulate = 0.0;
    double ba = 0.0;
    double filter = 0.0;
    double retriangulate = 0.0;
  } seconds;
};

// The whole loop. Next view = the unregistered camera with the most 2D-3D correspondences
// (>= min_correspondences; ties: lower index). A failed PnP is retried after the next BA. BA runs
// after the initial pair, after every `ba_every_n_cameras` registrations (if > 0; local or global,
// see ba_local_cameras), and globally whenever no candidate is left while something changed since
// the last global BA. (Without the early BAs the
// near-affine bunny views drift into the depth-reversed solution: the third camera is already
// 26 degrees off.) after each BA: filter_points, then triangulate the tracks
// still without a point. `ba` may be empty (no BA). Throws sfm::GeometryError if no initial pair.
Reconstruction reconstruct_incremental(const std::vector<matching::PairMatches>& pairs,
                                       const std::vector<features::Features>& features,
                                       const TrackIndex& index,
                                       const std::vector<Eigen::Matrix3d>& K,
                                       const IncrementalParams& p, const BundleAdjuster& ba,
                                       core::Logger* log, IncrementalReport* report = nullptr,
                                       const ReconObserver& observer = {});

}  // namespace sfm::recon
