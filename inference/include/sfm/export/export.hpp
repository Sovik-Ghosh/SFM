#pragma once
// Reconstruction export: PLY point cloud and COLMAP text model.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::exporter {

// ASCII PLY: one vertex per reconstructed point, "x y z red green blue" (gray as r = g = b).
// gray: per track, ignored where the track has no point. Throws IoError if the file cannot be
// written.
void write_ply(const std::filesystem::path& file, const recon::Reconstruction& rec,
               const std::vector<std::uint8_t>& gray);

// COLMAP text model (cameras.txt, images.txt, points3D.txt) in `dir` (created). One PINHOLE camera
// per registered image (fx, fy, cx, cy from K; K's skew is not representable and is dropped —
// it is < 1e-3 of fx on bunny). Image ids = index + 1, point ids = track + 1; POINTS2D lists every
// keypoint of the image (POINT3D_ID -1 where untracked or unreconstructed). names[i] = file name.
void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names,
                       const std::vector<std::uint8_t>& gray);

}  // namespace sfm::exporter
