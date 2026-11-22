#pragma once
// SceneSnapshot -> vertex arrays for drawing (points, camera frustums as line pairs), in the
// reconstruction's world frame. No OpenGL here: the renderer uploads these arrays as they are.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "sfm/eval/pose.hpp"
#include "sfm/pipeline/progress.hpp"
#include "sfm/viewer/colormap.hpp"

namespace sfm::viewer {

struct Vertex {
  float x = 0;
  float y = 0;
  float z = 0;
  std::uint8_t r = 0;
  std::uint8_t g = 0;
  std::uint8_t b = 0;
  std::uint8_t a = 255;
};

enum class ColorMode : std::uint8_t { kGray, kHeight, kError, kTrackLength };

struct SceneOptions {
  ColorMode color = ColorMode::kGray;
  bool frustums = true;
  float frustum_size = 0.4F;  // frustum depth as a fraction of the scene radius
  int image_width = 1024;     // image size for the frustum aspect (all images share it)
  int image_height = 768;
  int selected_camera = -1;  // drawn highlighted (image index)
  // Physical camera per image (CameraGroups::group): frustums in camera_group_color; empty =
  // coloured by registration order.
  std::vector<int> camera_group;
  // Ground-truth poses already in the reconstruction's frame (see gt_align.hpp); drawn in gray,
  // with a line from each true to each estimated centre. Empty = not drawn.
  std::vector<eval::Pose> ground_truth;
};

// A distinct colour per physical camera (8, then repeating), readable on the black background.
Rgb camera_group_color(int group);

struct SceneGeometry {
  std::vector<Vertex> points;
  std::vector<Vertex> lines;  // pairs of vertices
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float radius = 1.0F;         // 95 % of the points lie within this distance of center
  float camera_radius = 0.0F;  // largest distance of a registered camera centre from center
  Eigen::Vector3f up = Eigen::Vector3f::UnitZ();    // mean image "up" (-y axis) of the cameras
  std::array<float, 2> color_range = {0.0F, 1.0F};  // value range of the colour mode
  std::vector<Eigen::Vector3f> camera_centres;      // per image; NaN where not registered
};

// The four corners (in world coordinates) of the image plane at depth `depth` in front of camera
// (K, pose): K^-1 (u, v, 1) depth for (0, 0), (w, 0), (w, h), (0, h), mapped to the world.
std::array<Eigen::Vector3d, 4> frustum_corners(const Eigen::Matrix3d& K, const eval::Pose& pose,
                                               int width, int height, double depth);

SceneGeometry build_scene(const pipeline::SceneSnapshot& s, const SceneOptions& o);

// Ground grid: (2 cells + 1) lines in each direction, spacing radius / cells * 2, in the plane
// through center - radius * up, perpendicular to up (line pairs, dim gray).
std::vector<Vertex> grid_lines(const Eigen::Vector3f& center, float radius,
                               const Eigen::Vector3f& up, int cells = 10);

// Colour of camera k of n in registration order (viridis from first to last).
Rgb camera_color(int k, int n);

}  // namespace sfm::viewer
