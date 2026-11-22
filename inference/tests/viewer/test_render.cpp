// The renderer draws where the orbit camera says: a red point and a green line rendered off-screen
// land on the pixels predicted by project_to_pixel. Needs a display (skips without one).
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <GLFW/glfw3.h>

#include "sfm/viewer/orbit_camera.hpp"

#include "renderer.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

namespace {

class RenderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("DISPLAY") == nullptr || glfwInit() == 0) GTEST_SKIP() << "no display";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(64, 64, "render test", nullptr, nullptr);
    if (window_ == nullptr) GTEST_SKIP() << "no OpenGL 3.3 context";
    glfwMakeContextCurrent(window_);
  }
  void TearDown() override {
    if (window_ != nullptr) glfwDestroyWindow(window_);
    glfwTerminate();
  }
  GLFWwindow* window_ = nullptr;
};

std::array<int, 3> pixel(const std::vector<std::uint8_t>& rgb, int w, int x, int y) {
  const std::size_t i =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)) * 3;
  return {rgb[i], rgb[i + 1], rgb[i + 2]};
}

}  // namespace

TEST_F(RenderTest, PointAndLineLandOnThePredictedPixels) {
  constexpr int kW = 320;
  constexpr int kH = 240;
  gl::Framebuffer fb(kW, kH);
  gl::Renderer r;
  OrbitCamera cam;
  cam.fit(Eigen::Vector3f::Zero(), 2.0F);
  cam.rotate(0.4F, 0.1F);
  const Eigen::Matrix4f vp = cam.projection(static_cast<float>(kW) / kH) * cam.view();

  const Eigen::Vector3f red(0.6F, -0.3F, 0.4F);
  r.set_points({Vertex{red.x(), red.y(), red.z(), 255, 0, 0, 255}});
  const Eigen::Vector3f a(-1.0F, 0.8F, -0.5F);
  const Eigen::Vector3f b(-0.2F, 1.2F, 0.6F);
  r.set_lines(
      {Vertex{a.x(), a.y(), a.z(), 0, 255, 0, 255}, Vertex{b.x(), b.y(), b.z(), 0, 255, 0, 255}});
  fb.bind();
  r.draw(vp, 9.0F, kW, kH);
  const auto rgb = fb.read_rgb();
  gl::Framebuffer::bind_default();

  Eigen::Vector2f p;
  ASSERT_TRUE(project_to_pixel(vp, red, kW, kH, p));
  const auto c = pixel(rgb, kW, static_cast<int>(p.x()), static_cast<int>(p.y()));
  EXPECT_EQ(c, (std::array<int, 3>{255, 0, 0})) << p.transpose();

  // Midpoint of the line (projective midpoint in screen space ~ projection of the 3D midpoint for
  // this short segment; allow 1 px).
  Eigen::Vector2f m;
  ASSERT_TRUE(project_to_pixel(vp, 0.5F * (a + b), kW, kH, m));
  bool green = false;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const auto g = pixel(rgb, kW, static_cast<int>(m.x()) + dx, static_cast<int>(m.y()) + dy);
      green = green || (g[1] > 200 && g[0] < 60);
    }
  }
  EXPECT_TRUE(green) << m.transpose();

  // Background elsewhere: the clear colour, jet black (the approved design).
  EXPECT_EQ(pixel(rgb, kW, 2, 2), (std::array<int, 3>{0, 0, 0}));
}

TEST_F(RenderTest, ViewportDrawClearsOnlyItsRectangle) {
  constexpr int kW = 200;
  constexpr int kH = 100;
  gl::Framebuffer fb(kW, kH);
  gl::Renderer r;
  r.set_background(0.0F, 0.0F, 1.0F);
  fb.bind();
  gl::clear_target(kW, kH);  // all black
  // Right half only (GL coordinates: origin bottom-left).
  r.draw(Eigen::Matrix4f::Identity(), 1.0F, kW / 2, 0, kW / 2, kH);
  const auto rgb = fb.read_rgb();
  EXPECT_EQ(pixel(rgb, kW, 10, 50), (std::array<int, 3>{0, 0, 0}));     // left: untouched
  EXPECT_EQ(pixel(rgb, kW, 150, 50), (std::array<int, 3>{0, 0, 255}));  // right: cleared
  EXPECT_EQ(pixel(rgb, kW, 99, 0), (std::array<int, 3>{0, 0, 0}));      // the edge column
  EXPECT_EQ(pixel(rgb, kW, 100, 99), (std::array<int, 3>{0, 0, 255}));
}

TEST_F(RenderTest, WipeDrawsTwoRunsIntoTwoClipsOfOneView) {
  constexpr int kW = 200;
  constexpr int kH = 100;
  gl::Framebuffer fb(kW, kH);
  gl::Renderer a;
  gl::Renderer b;
  a.set_background(1.0F, 0.0F, 0.0F);
  b.set_background(0.0F, 0.0F, 1.0F);
  fb.bind();
  gl::clear_target(kW, kH);
  const gl::Renderer::Rect view{0, 0, kW, kH};
  a.draw(Eigen::Matrix4f::Identity(), 1.0F, view, gl::Renderer::Rect{0, 0, 60, kH}, true);
  b.draw(Eigen::Matrix4f::Identity(), 1.0F, view, gl::Renderer::Rect{60, 0, kW - 60, kH}, true);
  const auto rgb = fb.read_rgb();
  EXPECT_EQ(pixel(rgb, kW, 59, 50), (std::array<int, 3>{255, 0, 0}));  // A left of the divider
  EXPECT_EQ(pixel(rgb, kW, 60, 50), (std::array<int, 3>{0, 0, 255}));  // B right of it
  // Overlay: drawing without clearing keeps what is there.
  a.draw(Eigen::Matrix4f::Identity(), 1.0F, view, view, false);
  EXPECT_EQ(pixel(fb.read_rgb(), kW, 150, 50), (std::array<int, 3>{0, 0, 255}));
}
