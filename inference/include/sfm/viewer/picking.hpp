#pragma once
// Selecting a camera with the mouse: the camera whose centre projects nearest to the cursor
// (within max_px), nearer cameras winning ties.

#include <vector>

#include <Eigen/Core>

#include "sfm/viewer/scene.hpp"

namespace sfm::viewer {

// centres: per image (NaN = not registered). Returns the image index, or -1.
int pick_camera(const std::vector<Eigen::Vector3f>& centres, const Eigen::Matrix4f& view_proj,
                int width, int height, const Eigen::Vector2f& cursor, float max_px = 12.0F);

// Hovering a point: among the points whose projection lies within max_px of the cursor, the
// front-most one (smallest clip-space w: the visible surface of a dense cloud); equal depth ->
// the nearer on screen. Returns the index into `points`, or -1.
int pick_point(const std::vector<Vertex>& points, const Eigen::Matrix4f& view_proj, int width,
               int height, const Eigen::Vector2f& cursor, float max_px = 6.0F);

}  // namespace sfm::viewer
