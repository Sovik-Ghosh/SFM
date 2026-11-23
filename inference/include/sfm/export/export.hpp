#pragma once
// Reconstruction export: PLY point cloud and COLMAP text model.

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/features/features.hpp"
#include "sfm/recon/distortion.hpp"
#include "sfm/recon/reconstruction.hpp"

namespace sfm::exporter {

// Colour of a point: red, green, blue.
using Rgb = std::array<std::uint8_t, 3>;

// gray -> {g, g, g} per element.
std::vector<Rgb> gray_to_rgb(const std::vector<std::uint8_t>& gray);

// ASCII PLY: one vertex per reconstructed point, "x y z red green blue" (gray as r = g = b).
// gray: per track, ignored where the track has no point. Throws IoError if the file cannot be
// written.
void write_ply(const std::filesystem::path& file, const recon::Reconstruction& rec,
               const std::vector<std::uint8_t>& gray);
// The same with a colour per track (points without a colour are written gray, 128).
void write_ply(const std::filesystem::path& file, const recon::Reconstruction& rec,
               const std::vector<Rgb>& colours);

// COLMAP text model (cameras.txt, images.txt, points3D.txt) in `dir` (created). One PINHOLE camera
// per registered image (fx, fy, cx, cy from K; K's skew is not representable and is dropped —
// it is < 1e-3 of fx on bunny), WIDTH x HEIGHT = the image size. Image ids = index + 1, point ids =
// track + 1; POINTS2D lists every keypoint of the image (POINT3D_ID -1 where untracked or
// unreconstructed). names[i] = file name.
// Pixel convention: here a pixel's centre is at integer coordinates (features/pyramid.hpp); in
// COLMAP the top-left pixel's centre is (0.5, 0.5). cx, cy and every POINTS2D are written + 0.5.
void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names, int width, int height,
                       const std::vector<std::uint8_t>& gray);
void write_colmap_text(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                       const recon::TrackIndex& index,
                       const std::vector<features::Features>& features,
                       const std::vector<std::string>& names, int width, int height,
                       const std::vector<Rgb>& colours);

// Both COLMAP formats in `dir`: the text model above plus cameras.bin, images.bin, points3D.bin
// (COLMAP's binary format; same content). Tools that find both read the binary files.
// With distortion (dist.any()): camera model OPENCV (fx, fy, cx, cy, k1, k2, 0, 0) and `features`
// must hold the recorded (distorted) keypoints.
void write_colmap(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                  const recon::TrackIndex& index, const std::vector<features::Features>& features,
                  const std::vector<std::string>& names, int width, int height,
                  const std::vector<Rgb>& colours, const recon::Radial& dist = {});

// A scene folder that Gaussian-splatting and NeRF tools open directly (Inria 3DGS, nerfstudio's
// colmap dataparser, gsplat): <dir>/images/<name> and <dir>/sparse/0/ (cameras, images,
// points3D; .txt and .bin).
// JPEG and PNG images are copied as they are; PPM/PGM ones are converted to PNG (name.png). With
// distortion, every image is resampled to the pinhole camera (undistorted, PNG, same size and K)
// and `features` must be the undistorted keypoints: these tools read only pinhole cameras.
// images[i] = the source file of image i. Throws IoError on a write or conversion failure.
void write_colmap_scene(const std::filesystem::path& dir, const recon::Reconstruction& rec,
                        const recon::TrackIndex& index,
                        const std::vector<features::Features>& features,
                        const std::vector<std::filesystem::path>& images, int width, int height,
                        const std::vector<Rgb>& colours, const recon::Radial& dist = {});

}  // namespace sfm::exporter
