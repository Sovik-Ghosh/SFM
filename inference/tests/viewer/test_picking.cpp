#include <gtest/gtest.h>

#include <limits>

#include "sfm/viewer/orbit_camera.hpp"
#include "sfm/viewer/picking.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(Picking, NearestProjectedCentreWithinTolerance) {
  OrbitCamera c;
  c.fit(Eigen::Vector3f::Zero(), 2.0F);
  const Eigen::Matrix4f vp = c.projection(1.0F) * c.view();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<Eigen::Vector3f> centres = {Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(-1, 0, 0),
                                          Eigen::Vector3f(nan, nan, nan), Eigen::Vector3f(0, 0, 1)};
  for (int i : {0, 1, 3}) {
    Eigen::Vector2f px;
    ASSERT_TRUE(project_to_pixel(vp, centres[static_cast<std::size_t>(i)], 800, 800, px));
    EXPECT_EQ(pick_camera(centres, vp, 800, 800, px + Eigen::Vector2f(3, -2)), i);
  }
  EXPECT_EQ(pick_camera(centres, vp, 800, 800, Eigen::Vector2f(2, 2)), -1);  // empty corner
  // Two centres on the same ray: the nearer one wins.
  const Eigen::Vector3f near = c.eye() + 0.3F * (c.target() - c.eye());
  const Eigen::Vector3f far = c.target();
  const std::vector<Eigen::Vector3f> ray = {far, near};
  EXPECT_EQ(pick_camera(ray, vp, 800, 800, Eigen::Vector2f(400, 400)), 1);
}

TEST(PickPoint, FrontMostPointUnderTheCursorWins) {
  OrbitCamera cam;
  cam.fit(Eigen::Vector3f::Zero(), 2.0F);
  const int w = 400;
  const int h = 300;
  const Eigen::Matrix4f vp = cam.projection(static_cast<float>(w) / h) * cam.view();
  const Eigen::Vector3f eye = cam.eye();
  const Eigen::Vector3f dir = (Eigen::Vector3f::Zero() - eye).normalized();
  // Three points on the line of sight through the target (front, middle, back) and one off to
  // the side.
  std::vector<Vertex> pts;
  for (float t : {0.5F, 0.0F, -0.5F}) {
    const Eigen::Vector3f p = -t * dir;  // t > 0: towards the eye
    pts.push_back(Vertex{p.x(), p.y(), p.z()});
  }
  pts.push_back(Vertex{1.5F, 1.5F, 0.0F});
  Eigen::Vector2f centre;
  ASSERT_TRUE(project_to_pixel(vp, Eigen::Vector3f::Zero(), w, h, centre));
  EXPECT_EQ(pick_point(pts, vp, w, h, centre), 0);  // the front one hides the others
  EXPECT_EQ(pick_point(pts, vp, w, h, centre + Eigen::Vector2f(3, 2)), 0);    // within 6 px
  EXPECT_EQ(pick_point(pts, vp, w, h, centre + Eigen::Vector2f(40, 0)), -1);  // empty space
  Eigen::Vector2f side;
  ASSERT_TRUE(project_to_pixel(vp, Eigen::Vector3f(1.5F, 1.5F, 0.0F), w, h, side));
  EXPECT_EQ(pick_point(pts, vp, w, h, side), 3);
  EXPECT_EQ(pick_point({}, vp, w, h, centre), -1);
  // Behind the camera: never picked.
  const Eigen::Vector3f behind = eye - 3.0F * dir;
  EXPECT_EQ(pick_point({Vertex{behind.x(), behind.y(), behind.z()}}, vp, w, h, centre), -1);
}
