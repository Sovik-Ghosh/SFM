#include "sfm/viewer/orbit_camera.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>

namespace sfm::viewer {

namespace {

// Orthonormal frame (e1, e2) of the plane perpendicular to up.
void plane_frame(const Eigen::Vector3f& up, Eigen::Vector3f& e1, Eigen::Vector3f& e2) {
  const Eigen::Vector3f a =
      std::fabs(up.x()) < 0.9F ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitY();
  e1 = (a - a.dot(up) * up).normalized();
  e2 = up.cross(e1);
}

constexpr float kPitchLimit = 89.0F * 3.14159265F / 180.0F;

}  // namespace

void OrbitCamera::set_up(const Eigen::Vector3f& up) {
  if (up.norm() > 1e-9F) up_ = up.normalized();
}

void OrbitCamera::fit(const Eigen::Vector3f& center, float radius) {
  target_ = center;
  radius_ = std::max(radius, 1e-6F);
  distance_ = radius_ / std::sin(0.5F * fov_y_) * 1.1F;
}

void OrbitCamera::rotate(float d_yaw, float d_pitch) {
  yaw_ += d_yaw;
  pitch_ = std::clamp(pitch_ + d_pitch, -kPitchLimit, kPitchLimit);
}

void OrbitCamera::zoom(float factor) {
  distance_ = std::clamp(distance_ * factor, 1e-3F * radius_, 1e6F * radius_);
}

void OrbitCamera::pan(float dx, float dy) {
  const Eigen::Vector3f fwd = (target_ - eye()).normalized();
  const Eigen::Vector3f right = fwd.cross(up_).normalized();
  const Eigen::Vector3f cam_up = right.cross(fwd);
  target_ += (right * dx + cam_up * dy) * distance_;
}

void OrbitCamera::look_from(const Eigen::Vector3f& eye) {
  const Eigen::Vector3f d = eye - target_;
  const float n = d.norm();
  if (n < 1e-9F) return;
  Eigen::Vector3f e1;
  Eigen::Vector3f e2;
  plane_frame(up_, e1, e2);
  distance_ = std::clamp(n, 1e-3F * radius_, 1e6F * radius_);
  pitch_ =
      std::clamp(std::asin(std::clamp(d.dot(up_) / n, -1.0F, 1.0F)), -kPitchLimit, kPitchLimit);
  yaw_ = std::atan2(d.dot(e2), d.dot(e1));
}

void OrbitCamera::set_angles(float yaw, float pitch) {
  yaw_ = yaw;
  pitch_ = std::clamp(pitch, -kPitchLimit, kPitchLimit);
}

Eigen::Vector3f OrbitCamera::eye() const {
  // Orthonormal frame with up_ as the third axis.
  Eigen::Vector3f e1;
  Eigen::Vector3f e2;
  plane_frame(up_, e1, e2);
  const Eigen::Vector3f dir =
      std::cos(pitch_) * (std::cos(yaw_) * e1 + std::sin(yaw_) * e2) + std::sin(pitch_) * up_;
  return target_ + distance_ * dir;
}

Eigen::Matrix4f OrbitCamera::view() const { return look_at(eye(), target_, up_); }

Eigen::Matrix4f OrbitCamera::projection(float aspect) const {
  const float z_near = std::max(distance_ - 4.0F * radius_, distance_ * 1e-3F);
  const float z_far = distance_ + 4.0F * radius_;
  return perspective(fov_y_, aspect, z_near, z_far);
}

Eigen::Matrix4f look_at(const Eigen::Vector3f& eye, const Eigen::Vector3f& target,
                        const Eigen::Vector3f& up) {
  const Eigen::Vector3f f = (target - eye).normalized();
  const Eigen::Vector3f s = f.cross(up).normalized();
  const Eigen::Vector3f u = s.cross(f);
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<1, 3>(0, 0) = s.transpose();
  m.block<1, 3>(1, 0) = u.transpose();
  m.block<1, 3>(2, 0) = -f.transpose();
  m(0, 3) = -s.dot(eye);
  m(1, 3) = -u.dot(eye);
  m(2, 3) = f.dot(eye);
  return m;
}

Eigen::Matrix4f perspective(float fov_y, float aspect, float z_near, float z_far) {
  const float f = 1.0F / std::tan(0.5F * fov_y);
  Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
  m(0, 0) = f / aspect;
  m(1, 1) = f;
  m(2, 2) = (z_far + z_near) / (z_near - z_far);
  m(2, 3) = 2.0F * z_far * z_near / (z_near - z_far);
  m(3, 2) = -1.0F;
  return m;
}

bool project_to_pixel(const Eigen::Matrix4f& view_proj, const Eigen::Vector3f& p, int w, int h,
                      Eigen::Vector2f& px) {
  const Eigen::Vector4f c = view_proj * p.homogeneous();
  if (!(c.w() > 0.0F)) return false;
  const float x = c.x() / c.w();
  const float y = c.y() / c.w();
  px = {(x + 1.0F) * 0.5F * static_cast<float>(w), (1.0F - y) * 0.5F * static_cast<float>(h)};
  return true;
}

}  // namespace sfm::viewer
