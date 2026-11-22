// Viewer V5 kernels without OpenGL: the colours they write into device memory equal the CPU scene
// builder's, byte for byte, in every mode (runs on the GPU CI runner; no display needed).
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "sfm/core/cuda_check.h"
#include "sfm/viewer/scene.hpp"

#include "interop.hpp"
#include "snapshot_util.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

namespace {

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

}  // namespace

TEST(InteropKernels, DeviceColoursEqualTheCpuSceneBuilder) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "no CUDA device";
  constexpr int kN = 50000;
  sfm::pipeline::SceneSnapshot s = test::synthetic_snapshot(6, 10);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> pos(-15.0, 15.0);
  std::uniform_real_distribution<float> err(0.0F, 3.0F);
  std::uniform_int_distribution<int> small(0, 255);
  s.points.clear();
  s.point_gray.clear();
  s.point_error_px.clear();
  s.point_track_length.clear();
  for (int k = 0; k < kN; ++k) {
    s.points.emplace_back(pos(rng), pos(rng), pos(rng));
    s.point_gray.push_back(static_cast<std::uint8_t>(small(rng)));
    s.point_error_px.push_back(err(rng));
    s.point_track_length.push_back(2 + small(rng) % 30);
  }
  SceneOptions o;
  o.frustums = false;
  const std::vector<Vertex> start = build_scene(s, o).points;  // gray
  Vertex* d = nullptr;
  CHECK(cudaMalloc(reinterpret_cast<void**>(&d), start.size() * sizeof(Vertex)));
  CHECK(cudaMemcpy(d, start.data(), start.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
  gl::PointColorsCuda cuda;
  cuda.set_attributes(s.point_gray, s.point_error_px, s.point_track_length);
  for (ColorMode mode :
       {ColorMode::kHeight, ColorMode::kError, ColorMode::kTrackLength, ColorMode::kGray}) {
    SCOPED_TRACE(static_cast<int>(mode));
    o.color = mode;
    const SceneGeometry cpu = build_scene(s, o);
    const auto range =
        cuda.recolor_device(d, kN, static_cast<int>(mode), {cpu.up.x(), cpu.up.y(), cpu.up.z()});
    EXPECT_EQ(range[0], cpu.color_range[0]);
    EXPECT_EQ(range[1], cpu.color_range[1]);
    std::vector<Vertex> gpu(start.size());
    CHECK(cudaMemcpy(gpu.data(), d, gpu.size() * sizeof(Vertex), cudaMemcpyDeviceToHost));
    EXPECT_EQ(count_different(gpu, cpu.points), 0);
  }
  CHECK(cudaFree(d));
}
