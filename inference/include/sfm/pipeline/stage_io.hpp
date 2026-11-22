#pragma once
// Save / load of every stage output as .npy (readable from NumPy), so any stage can be re-run
// alone and the Python tools can compare stage by stage. Layout under <dir>:
//   features/NNNN_{xy,angle,response,octave,desc}.npy   per image (NNNN = position in the run)
//   matching/{pairs,idx,F}.npy    pairs (P, 3) i32 = i, j, inlier count; idx (M, 2) i32 all pairs'
//                                 inliers concatenated; F (P, 3, 3) f64
//   tracks/obs.npy                (O, 3) i32 = track, image, keypoint
//   reconstruction/{R,t,registered,order,points,has_point,obs_outlier}.npy (+ report.npy, written
//   by the
//                                 pipeline: pnp attempts, pnp failures, BA runs)

#include <filesystem>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/matching/matches.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::pipeline {

void save_features(const std::filesystem::path& dir, const std::vector<features::Features>& f);
std::vector<features::Features> load_features(const std::filesystem::path& dir, std::size_t images);

void save_matches(const std::filesystem::path& dir, const std::vector<matching::PairMatches>& p);
std::vector<matching::PairMatches> load_matches(const std::filesystem::path& dir);

void save_tracks(const std::filesystem::path& dir, const matching::Tracks& t);
matching::Tracks load_tracks(const std::filesystem::path& dir);

void save_reconstruction(const std::filesystem::path& dir, const recon::Reconstruction& r);
// K is not stored (it comes from the calibration): cameras[i].K = K[i].
recon::Reconstruction load_reconstruction(const std::filesystem::path& dir,
                                          const std::vector<Eigen::Matrix3d>& K);

}  // namespace sfm::pipeline
