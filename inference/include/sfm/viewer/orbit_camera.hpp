#pragma once
// Orbit camera for the 3D view: looks at a target from a distance, rotated by yaw (around the
// world up axis) and pitch. Produces OpenGL-convention matrices (right-handed view space looking
// down -z, clip space z in [-1, 1]), column-major in Eigen.

#include <Eigen/Core>

namespace sfm::viewer {

class OrbitCamera {
 public:
  void set_up(const Eigen::Vector3f& up);  // world up (normalised internally)
  void set_target(const Eigen::Vector3f& t) { target_ = t; }
  // Frame a sphere (center, radius): target = center, distance so the sphere fits the vertical
  // field of view, yaw / pitch kept.
  void fit(const Eigen::Vector3f& center, float radius);

  void rotate(float d_yaw, float d_pitch);  // radians; pitch clamped to (-89, 89) degrees
  void zoom(float factor);                  // distance *= factor, clamped to [1e-3, 1e6] radii
  void pan(float dx, float dy);             // in units of the distance (screen right / up)
  // Moves the eye to `eye` (target kept): distance, yaw and pitch from eye - target (pitch
  // clamped like rotate). Used by "look through camera".
  void look_from(const Eigen::Vector3f& eye);
  // View presets (radians): pitch clamped to (-89, 89) degrees.
  void set_angles(float yaw, float pitch);
  float yaw() const { return yaw_; }
  float pitch() const { return pitch_; }

  Eigen::Vector3f eye() const;
  const Eigen::Vector3f& target() const { return target_; }
  float distance() const { return distance_; }
  float fov_y() const { return fov_y_; }

  Eigen::Matrix4f view() const;
  Eigen::Matrix4f projection(float aspect) const;  // near / far from the distance and radius

 private:
  Eigen::Vector3f up_ = Eigen::Vector3f::UnitZ();
  Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
  float distance_ = 5.0F;
  float radius_ = 1.0F;
  float yaw_ = 0.6F;
  float pitch_ = 0.35F;
  float fov_y_ = 0.8F;  // radians (~46 degrees)
};

// OpenGL look-at and perspective (gluLookAt / gluPerspective conventions).
Eigen::Matrix4f look_at(const Eigen::Vector3f& eye, const Eigen::Vector3f& target,
                        const Eigen::Vector3f& up);
Eigen::Matrix4f perspective(float fov_y, float aspect, float z_near, float z_far);

// World point -> pixel (x right, y down, origin top-left) for a viewport w x h; false if behind
// the camera.
bool project_to_pixel(const Eigen::Matrix4f& view_proj, const Eigen::Vector3f& p, int w, int h,
                      Eigen::Vector2f& px);

}  // namespace sfm::viewer
