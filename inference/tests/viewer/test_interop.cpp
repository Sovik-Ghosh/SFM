// Viewer V5: the CUDA kernels write into the OpenGL vertex buffer exactly the colours (and range)
// the CPU scene builder computes, for every colour mode. Needs a display and a CUDA device (skips
// without them).
#define GL_GLEXT_PROTOTYPES
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include "sfm/viewer/scene.hpp"

#include "interop.hpp"
#include "renderer.hpp"
#include "snapshot_util.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

namespace {

class InteropTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("DISPLAY") == nullptr || glfwInit() == 0) GTEST_SKIP() << "no display";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(64, 64, "interop test", nullptr, nullptr);
    if (window_ == nullptr) GTEST_SKIP() << "no OpenGL 3.3 context";
    glfwMakeContextCurrent(window_);
    if (!gl::PointColorsCuda::available()) GTEST_SKIP() << "no CUDA device for this GL context";
  }
  void TearDown() override {
    if (window_ != nullptr) glfwDestroyWindow(window_);
    glfwTerminate();
  }
  GLFWwindow* window_ = nullptr;
};

// The synthetic cameras with many random points and attributes (ties in the values on purpose:
// gray and track length repeat, which exercises the quantile picks).
sfm::pipeline::SceneSnapshot big_snapshot(int n) {
  sfm::pipeline::SceneSnapshot s = test::synthetic_snapshot(6, 10);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> pos(-15.0, 15.0);
  std::uniform_real_distribution<float> err(0.0F, 3.0F);
  std::uniform_int_distribution<int> gray(0, 255);
  std::uniform_int_distribution<int> track(2, 40);
  s.points.clear();
  s.point_gray.clear();
  s.point_error_px.clear();
  s.point_track_length.clear();
  for (int k = 0; k < n; ++k) {
    s.points.emplace_back(pos(rng), pos(rng), pos(rng));
    s.point_gray.push_back(static_cast<std::uint8_t>(gray(rng)));
    s.point_error_px.push_back(err(rng));
    s.point_track_length.push_back(track(rng));
  }
  return s;
}

// Vertices that differ (position or colour), field by field.
int count_different(const std::vector<Vertex>& a, const std::vector<Vertex>& b) {
  if (a.size() != b.size()) return -1;
  int n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    const Vertex& p = a[k];
    const Vertex& q = b[k];
    const bool same = p.x == q.x && p.y == q.y && p.z == q.z && p.r == q.r && p.g == q.g &&
                      p.b == q.b && p.a == q.a;
    n += same ? 0 : 1;
  }
  return n;
}

std::vector<Vertex> read_back(unsigned int vbo, int n) {
  std::vector<Vertex> v(static_cast<std::size_t>(n));
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glGetBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(v.size() * sizeof(Vertex)),
                     v.data());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  return v;
}

}  // namespace

TEST_F(InteropTest, GpuColoursEqualTheCpuSceneBuilderInEveryMode) {
  constexpr int kN = 200000;
  const auto s = big_snapshot(kN);
  SceneOptions o;
  o.frustums = false;
  o.color = ColorMode::kGray;
  const SceneGeometry gray = build_scene(s, o);  // the buffer starts with the gray colours
  gl::Renderer r;
  r.set_points(gray.points);
  ASSERT_EQ(r.point_count(), kN);
  gl::PointColorsCuda cuda;
  cuda.attach(r.points_vbo(), r.point_count());
  cuda.set_attributes(s.point_gray, s.point_error_px, s.point_track_length);
  for (ColorMode mode :
       {ColorMode::kHeight, ColorMode::kError, ColorMode::kTrackLength, ColorMode::kGray}) {
    SCOPED_TRACE(static_cast<int>(mode));
    o.color = mode;
    const SceneGeometry cpu = build_scene(s, o);
    const auto range = cuda.recolor(static_cast<int>(mode), {cpu.up.x(), cpu.up.y(), cpu.up.z()});
    EXPECT_EQ(range[0], cpu.color_range[0]);
    EXPECT_EQ(range[1], cpu.color_range[1]);
    const auto gpu = read_back(r.points_vbo(), kN);
    EXPECT_EQ(count_different(gpu, cpu.points), 0) << "vertices that differ";
  }
}

TEST_F(InteropTest, ReattachAfterTheBufferIsReplaced) {
  const auto s = big_snapshot(1000);
  SceneOptions o;
  o.frustums = false;
  gl::Renderer r;
  gl::PointColorsCuda cuda;
  for (int round = 0; round < 3; ++round) {
    o.color = ColorMode::kGray;
    r.set_points(build_scene(s, o).points);  // re-specifies the buffer
    cuda.attach(r.points_vbo(), r.point_count());
    cuda.set_attributes(s.point_gray, s.point_error_px, s.point_track_length);
    o.color = ColorMode::kError;
    const SceneGeometry cpu = build_scene(s, o);
    cuda.recolor(2, {cpu.up.x(), cpu.up.y(), cpu.up.z()});
    const auto gpu = read_back(r.points_vbo(), r.point_count());
    EXPECT_EQ(count_different(gpu, cpu.points), 0) << round;
  }
}
