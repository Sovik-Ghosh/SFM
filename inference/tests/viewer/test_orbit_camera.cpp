#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "sfm/viewer/orbit_camera.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(OrbitCamera, ViewIsRigidAndLooksAtTheTarget) {
  OrbitCamera c;
  c.set_up(Eigen::Vector3f(0.1F, 0.2F, 1.0F));
  c.fit(Eigen::Vector3f(1, 2, 3), 2.0F);
  c.rotate(0.7F, 0.2F);
  const Eigen::Matrix4f v = c.view();
  const Eigen::Matrix3f r = v.topLeftCorner<3, 3>();
  EXPECT_LT((r * r.transpose() - Eigen::Matrix3f::Identity()).norm(), 1e-5F);
  EXPECT_NEAR(r.determinant(), 1.0F, 1e-5F);
  const Eigen::Vector4f t = v * Eigen::Vector4f(1, 2, 3, 1);  // the target
  EXPECT_NEAR(t.x(), 0.0F, 1e-4F);
  EXPECT_NEAR(t.y(), 0.0F, 1e-4F);
  EXPECT_NEAR(t.z(), -c.distance(), 1e-3F);
  const Eigen::Vector4f e = v * c.eye().homogeneous();
  EXPECT_LT(e.head<3>().norm(), 1e-4F);  // the eye is the origin of view space
}

TEST(OrbitCamera, PerspectiveMatchesTheFormula) {
  const Eigen::Matrix4f p = perspective(0.8F, 1.5F, 0.5F, 50.0F);
  const float f = 1.0F / std::tan(0.4F);
  EXPECT_NEAR(p(0, 0), f / 1.5F, 1e-6F);
  EXPECT_NEAR(p(1, 1), f, 1e-6F);
  // z = -near -> NDC -1, z = -far -> NDC +1.
  const Eigen::Vector4f n = p * Eigen::Vector4f(0, 0, -0.5F, 1);
  const Eigen::Vector4f fa = p * Eigen::Vector4f(0, 0, -50.0F, 1);
  EXPECT_NEAR(n.z() / n.w(), -1.0F, 1e-5F);
  EXPECT_NEAR(fa.z() / fa.w(), 1.0F, 1e-5F);
}

TEST(OrbitCamera, FitKeepsTheSphereInView) {
  OrbitCamera c;
  c.fit(Eigen::Vector3f(5, -2, 1), 3.0F);
  const float aspect = 1024.0F / 768.0F;
  const Eigen::Matrix4f vp = c.projection(aspect) * c.view();
  for (int k = 0; k < 200; ++k) {  // points on the sphere
    const float a = 0.13F * static_cast<float>(k);
    const float b = 0.071F * static_cast<float>(k);
    const Eigen::Vector3f p =
        Eigen::Vector3f(5, -2, 1) +
        3.0F * Eigen::Vector3f(std::cos(a) * std::cos(b), std::sin(a) * std::cos(b), std::sin(b));
    const Eigen::Vector4f h = vp * p.homogeneous();
    ASSERT_GT(h.w(), 0.0F);
    EXPECT_LE(std::fabs(h.x() / h.w()), 1.0F) << k;
    EXPECT_LE(std::fabs(h.y() / h.w()), 1.0F) << k;
    EXPECT_LE(std::fabs(h.z() / h.w()), 1.0F) << k;  // inside near / far
  }
  Eigen::Vector2f px;
  ASSERT_TRUE(project_to_pixel(vp, Eigen::Vector3f(5, -2, 1), 1024, 768, px));
  EXPECT_NEAR(px.x(), 512.0F, 1e-2F);  // the target is the viewport centre
  EXPECT_NEAR(px.y(), 384.0F, 1e-2F);
}

TEST(OrbitCamera, LimitsAndPan) {
  OrbitCamera c;
  c.fit(Eigen::Vector3f::Zero(), 1.0F);
  c.rotate(0.0F, 10.0F);  // far beyond 89 degrees
  const Eigen::Vector3f d = (c.eye() - c.target()).normalized();
  EXPECT_LT(d.z(), std::sin(89.5F * 3.14159265F / 180.0F));
  const float d0 = c.distance();
  c.zoom(1e-9F);
  EXPECT_NEAR(c.distance(), 1e-3F, 1e-6F);
  c.zoom(1e12F);
  EXPECT_NEAR(c.distance(), 1e6F, 1.0F);
  c.fit(Eigen::Vector3f::Zero(), 1.0F);
  EXPECT_NEAR(c.distance(), d0, 1e-5F);
  // Panning right moves the view right, so the old target appears left of centre.
  c.pan(0.1F, 0.0F);
  const Eigen::Vector4f old_target_in_view = c.view() * Eigen::Vector4f(0, 0, 0, 1);
  EXPECT_LT(old_target_in_view.x(), 0.0F);
}

TEST(OrbitCamera, BehindTheCameraIsNotProjected) {
  OrbitCamera c;
  c.fit(Eigen::Vector3f::Zero(), 1.0F);
  const Eigen::Matrix4f vp = c.projection(1.0F) * c.view();
  const Eigen::Vector3f behind = c.eye() + (c.eye() - c.target());
  Eigen::Vector2f px;
  EXPECT_FALSE(project_to_pixel(vp, behind, 100, 100, px));
}

TEST(OrbitCamera, LookFromPutsTheEyeThere) {
  OrbitCamera cam;
  cam.set_up(Eigen::Vector3f(0.2F, -0.1F, 1.0F));
  cam.fit(Eigen::Vector3f(1, 2, 3), 2.0F);
  for (const Eigen::Vector3f& eye : {Eigen::Vector3f(10, -4, 6), Eigen::Vector3f(-3, 8, 2)}) {
    cam.look_from(eye);
    EXPECT_LT((cam.eye() - eye).norm(), 1e-4F);
  }
  cam.set_angles(0.3F, 2.0F);  // pitch clamped to 89 degrees
  EXPECT_FLOAT_EQ(cam.yaw(), 0.3F);
  EXPECT_NEAR(cam.pitch(), 89.0F * 3.14159265F / 180.0F, 1e-6);
}
