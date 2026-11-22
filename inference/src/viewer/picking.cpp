#include "sfm/viewer/picking.hpp"

#include <cmath>
#include <limits>

#include <Eigen/Geometry>

#include "sfm/viewer/orbit_camera.hpp"

namespace sfm::viewer {

int pick_camera(const std::vector<Eigen::Vector3f>& centres, const Eigen::Matrix4f& view_proj,
                int width, int height, const Eigen::Vector2f& cursor, float max_px) {
  int best = -1;
  float best_d = std::numeric_limits<float>::infinity();
  float best_depth = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < centres.size(); ++i) {
    const Eigen::Vector3f& c = centres[i];
    if (!c.allFinite()) continue;
    Eigen::Vector2f px;
    if (!project_to_pixel(view_proj, c, width, height, px)) continue;
    const float d = (px - cursor).norm();
    if (d > max_px) continue;
    const float depth = (view_proj * c.homogeneous()).w();
    // Prefer the smaller screen distance; within 1 px, the nearer camera.
    if (d < best_d - 1.0F || (std::fabs(d - best_d) <= 1.0F && depth < best_depth)) {
      best = static_cast<int>(i);
      best_d = d;
      best_depth = depth;
    }
  }
  return best;
}

int pick_point(const std::vector<Vertex>& points, const Eigen::Matrix4f& view_proj, int width,
               int height, const Eigen::Vector2f& cursor, float max_px) {
  int best = -1;
  float best_depth = std::numeric_limits<float>::infinity();
  float best_d = std::numeric_limits<float>::infinity();
  const float w2 = 0.5F * static_cast<float>(width);
  const float h2 = 0.5F * static_cast<float>(height);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector4f c =
        view_proj * Eigen::Vector4f(points[i].x, points[i].y, points[i].z, 1.0F);
    if (c.w() <= 0.0F) continue;  // behind the camera
    // As project_to_pixel, inlined: this runs over every point on each mouse move.
    const float px = (c.x() / c.w() + 1.0F) * w2;
    const float py = (1.0F - c.y() / c.w()) * h2;
    const float d = std::hypot(px - cursor.x(), py - cursor.y());
    if (d > max_px) continue;
    if (c.w() < best_depth || (c.w() == best_depth && d < best_d)) {
      best = static_cast<int>(i);
      best_depth = c.w();
      best_d = d;
    }
  }
  return best;
}

}  // namespace sfm::viewer
