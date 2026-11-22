#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Geometry>

#include "sfm/viewer/scene.hpp"

#include "snapshot_util.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(Scene, FrustumCornersProjectOntoTheImageCorners) {
  const auto s = test::synthetic_snapshot();
  const sfm::eval::Pose pose{s.R[2], s.t[2]};
  const auto c = frustum_corners(s.K[2], pose, 1024, 768, 3.5);
  const Eigen::Vector2d want[4] = {{0, 0}, {1024, 0}, {1024, 768}, {0, 768}};
  for (int k = 0; k < 4; ++k) {
    const Eigen::Vector3d xc = pose.R * c[static_cast<std::size_t>(k)] + pose.t;
    EXPECT_NEAR(xc.z(), 3.5, 1e-9);  // at the requested depth
    const Eigen::Vector3d h = s.K[2] * xc;
    EXPECT_LT((h.head<2>() / h.z() - want[k]).norm(), 1e-9) << k;
  }
}

TEST(Scene, GeometryCountsColoursAndFrame) {
  auto s = test::synthetic_snapshot(6, 50);
  s.registered[4] = 0;  // one camera not registered
  s.order = {0, 1, 2, 3, 5};
  SceneOptions o;
  const SceneGeometry g = build_scene(s, o);
  ASSERT_EQ(g.points.size(), 50u);
  EXPECT_EQ(g.points[7].r, 35);  // gray mode: point_gray 7 * 5
  EXPECT_EQ(g.points[7].g, 35);
  EXPECT_EQ(g.lines.size(), 5u * 9u * 2u);  // per frustum: 4 rays, 4 edges, 1 up tick
  EXPECT_TRUE(std::isnan(g.camera_centres[4].x()));
  EXPECT_TRUE(g.camera_centres[0].allFinite());
  // Radius covers 95 % of the points.
  int inside = 0;
  for (const auto& p : s.points) {
    inside += (p.cast<float>() - g.center).norm() <= g.radius + 1e-5F ? 1 : 0;
  }
  EXPECT_GE(inside, 47);
  // Synthetic cameras are 60 units from the origin; the points within 10 * sqrt(3).
  EXPECT_GT(g.camera_radius, 40.0F);
  EXPECT_LT(g.camera_radius, 80.0F);
  // Synthetic cameras look at the origin with world +z up in the image: up ~ +z.
  EXPECT_GT(g.up.z(), 0.9F);

  o.color = ColorMode::kError;
  const SceneGeometry e = build_scene(s, o);
  EXPECT_LT(e.color_range[0], e.color_range[1]);
  EXPECT_EQ((Rgb{e.points[0].r, e.points[0].g, e.points[0].b}), viridis(0.0F));     // lowest error
  EXPECT_EQ((Rgb{e.points[49].r, e.points[49].g, e.points[49].b}), viridis(1.0F));  // highest

  o.frustums = false;
  EXPECT_TRUE(build_scene(s, o).lines.empty());
}

TEST(Scene, GroundTruthAndSelectionAddLines) {
  const auto s = test::synthetic_snapshot(4, 20);
  SceneOptions o;
  o.selected_camera = 2;
  const SceneGeometry g = build_scene(s, o);
  // The selected camera's lines are orange.
  bool orange = false;
  for (const auto& v : g.lines) orange = orange || (v.r == 255 && v.g == 80 && v.b == 40);
  EXPECT_TRUE(orange);
  for (std::size_t i = 0; i < 4; ++i) o.ground_truth.push_back({s.R[i], s.t[i]});
  const SceneGeometry gt = build_scene(s, o);
  EXPECT_EQ(gt.lines.size(),
            g.lines.size() + std::size_t{4} * (9 + 1) * 2);  // frustum + error line each
}

TEST(Scene, CameraColoursFollowRegistrationOrder) {
  EXPECT_EQ(camera_color(0, 5), viridis(0.0F));
  EXPECT_EQ(camera_color(4, 5), viridis(1.0F));
  EXPECT_EQ(camera_color(0, 1), viridis(1.0F));
}

TEST(Scene, GridLiesInAPlanePerpendicularToUp) {
  const Eigen::Vector3f up = Eigen::Vector3f(0.2F, 0.1F, 1.0F).normalized();
  const auto g = grid_lines(Eigen::Vector3f(1, 2, 3), 2.0F, up, 4);
  EXPECT_EQ(g.size(), 2u * 2u * 9u);  // 9 lines each way, 2 vertices per line
  const Eigen::Vector3f o = Eigen::Vector3f(1, 2, 3) - 2.0F * up;
  for (const auto& v : g) EXPECT_NEAR((Eigen::Vector3f(v.x, v.y, v.z) - o).dot(up), 0.0F, 1e-5F);
}

// Rig datasets (P9a): frustums take the colour of their physical camera; a size mismatch falls
// back to the registration order.
TEST(Scene, CameraGroupsColourTheFrustums) {
  const auto s = test::synthetic_snapshot(4, 20);
  SceneOptions o;
  o.camera_group = {0, 1, 0, 1};
  const SceneGeometry g = build_scene(s, o);
  ASSERT_EQ(g.lines.size(), 4u * 9u * 2u);
  const std::size_t per = std::size_t{9} * 2;  // vertices per frustum, cameras in s.order
  for (std::size_t k = 0; k < s.order.size(); ++k) {
    const auto i = static_cast<std::size_t>(s.order[k]);
    const Vertex& v = g.lines[k * per];
    EXPECT_EQ((Rgb{v.r, v.g, v.b}), camera_group_color(o.camera_group[i])) << k;
  }
  EXPECT_NE(camera_group_color(0), camera_group_color(1));
  EXPECT_EQ(camera_group_color(8), camera_group_color(0));  // the palette repeats
  o.camera_group = {0, 1};                                  // wrong size: order colours
  const SceneGeometry f = build_scene(s, o);
  EXPECT_EQ((Rgb{f.lines[0].r, f.lines[0].g, f.lines[0].b}),
            camera_color(0, static_cast<int>(s.order.size())));
}

// Some captures store images rotated 180 degrees (Middlebury's rings): their "up" vectors cancel
// the others in a mean, but the up axis is still found (signed by the majority).
TEST(Scene, UpSurvivesImagesRotated180Degrees) {
  auto s = test::synthetic_snapshot(8, 20);
  const Eigen::Matrix3d flip = Eigen::Vector3d(-1, -1, 1).asDiagonal();  // 180 deg about the view
  for (std::size_t i = 0; i < 3; ++i) {  // 3 of 8 flipped: the mean up shrinks, the axis stays
    s.R[i] = flip * s.R[i];
    s.t[i] = flip * s.t[i];
  }
  const SceneGeometry g = build_scene(s, SceneOptions{});
  EXPECT_GT(g.up.z(), 0.9F);
}
