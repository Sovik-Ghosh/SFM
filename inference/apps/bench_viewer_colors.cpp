// Viewer V5 benchmark: time from a colour-mode change to the vertex buffer holding the new colours,
// for N synthetic points: the CPU path (build_scene + upload of every vertex) against CUDA-OpenGL
// interop (two kernels + a sort on the device). Needs a display and a CUDA device.
//   bench_viewer_colors [points=1000000] [repeats=5]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include "sfm/eval/synthetic.hpp"
#include "sfm/viewer/scene.hpp"

#include "interop.hpp"
#include "renderer.hpp"

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 1000000;
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
  if (glfwInit() == 0) return 1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(64, 64, "bench", nullptr, nullptr);
  if (window == nullptr) return 1;
  glfwMakeContextCurrent(window);
  if (!sfm::viewer::gl::PointColorsCuda::available()) {
    std::fprintf(stderr, "no CUDA device for this OpenGL context\n");
    return 1;
  }
  // Synthetic cameras and n random points with attributes.
  sfm::eval::SyntheticConfig c;
  c.cameras = 8;
  c.points = 10;
  const auto scene = sfm::eval::make_synthetic_scene(c);
  sfm::pipeline::SceneSnapshot s;
  for (int i = 0; i < c.cameras; ++i) {
    s.ids.push_back(i);
    s.K.push_back(scene.K);
    s.R.push_back(scene.poses[static_cast<std::size_t>(i)].R);
    s.t.push_back(scene.poses[static_cast<std::size_t>(i)].t);
    s.registered.push_back(1);
    s.order.push_back(i);
  }
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> pos(-15.0, 15.0);
  std::uniform_real_distribution<float> err(0.0F, 3.0F);
  std::uniform_int_distribution<int> small(0, 255);
  for (int k = 0; k < n; ++k) {
    s.points.emplace_back(pos(rng), pos(rng), pos(rng));
    s.point_gray.push_back(static_cast<std::uint8_t>(small(rng)));
    s.point_error_px.push_back(err(rng));
    s.point_track_length.push_back(2 + small(rng) % 30);
  }
  sfm::viewer::SceneOptions o;
  o.frustums = false;
  sfm::viewer::gl::Renderer r;
  r.set_points(sfm::viewer::build_scene(s, o).points);
  sfm::viewer::gl::PointColorsCuda cuda;
  cuda.attach(r.points_vbo(), r.point_count());
  cuda.set_attributes(s.point_gray, s.point_error_px, s.point_track_length);
  const sfm::viewer::ColorMode modes[] = {
      sfm::viewer::ColorMode::kHeight, sfm::viewer::ColorMode::kError,
      sfm::viewer::ColorMode::kTrackLength, sfm::viewer::ColorMode::kGray};
  cuda.recolor(1, {0, 0, 1});  // warm-up (kernels, thrust allocations)
  glFinish();
  std::vector<double> cpu_ms;
  std::vector<double> gpu_ms;
  for (int rep = 0; rep < repeats; ++rep) {
    for (const auto mode : modes) {
      o.color = mode;
      auto t0 = std::chrono::steady_clock::now();
      const auto g = sfm::viewer::build_scene(s, o);
      r.set_points(g.points);
      glFinish();
      cpu_ms.push_back(ms_since(t0));
      cuda.attach(r.points_vbo(), r.point_count());  // the CPU path re-specified the buffer
      t0 = std::chrono::steady_clock::now();
      cuda.recolor(static_cast<int>(mode), {g.up.x(), g.up.y(), g.up.z()});
      glFinish();
      gpu_ms.push_back(ms_since(t0));
    }
  }
  std::printf("points %d, %d mode changes each\n", n, static_cast<int>(cpu_ms.size()));
  std::printf("CPU build_scene + upload: median %.1f ms (min %.1f)\n", median(cpu_ms),
              *std::min_element(cpu_ms.begin(), cpu_ms.end()));
  std::printf("CUDA-GL interop recolor:  median %.1f ms (min %.1f)\n", median(gpu_ms),
              *std::min_element(gpu_ms.begin(), gpu_ms.end()));
  std::printf("speed-up (medians): %.1fx\n", median(cpu_ms) / median(gpu_ms));
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
