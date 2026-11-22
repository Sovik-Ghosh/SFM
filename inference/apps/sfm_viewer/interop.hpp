#pragma once
// Viewer V5: point colours computed on the GPU and written straight into the OpenGL vertex buffer
// (CUDA-OpenGL interop). The per-point attributes stay on the device, so changing the colour mode
// is two kernels and a sort, with no vertex rebuild or upload. Colours are byte-identical to the
// CPU scene builder (sfm/viewer/colormap_hd.hpp). No Eigen here: this header is compiled by nvcc.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace sfm::viewer::gl {

class PointColorsCuda {
 public:
  // True when a CUDA device is visible and this OpenGL context can share buffers with it.
  static bool available();

  PointColorsCuda();
  ~PointColorsCuda();
  PointColorsCuda(const PointColorsCuda&) = delete;
  PointColorsCuda& operator=(const PointColorsCuda&) = delete;

  // Registers the vertex buffer (sfm::viewer::Vertex layout) holding `count` points; call again
  // after the buffer is re-specified (Renderer::set_points). Throws sfm::CudaError on failure.
  void attach(unsigned int vbo, int count);
  // Uploads the points' gray, error and track length (SceneSnapshot::point_gray, point_error_px,
  // point_track_length: same order as the buffer's vertices).
  void set_attributes(const std::vector<std::uint8_t>& gray, const std::vector<float>& error_px,
                      const std::vector<int>& track_length);
  // Writes the colours of `mode` (static_cast<int>(ColorMode)) into the buffer, with the world up
  // as in build_scene (SceneGeometry::up). Returns the value range used (the 5-95 % quantiles, as
  // SceneGeometry::color_range; {0, 1} for gray).
  std::array<float, 2> recolor(int mode, const std::array<float, 3>& up);
  // The same kernels on `n` vertices in ordinary device memory (no OpenGL): the GL path maps the
  // buffer and calls this. For tests and the sanitizers (initcheck cannot follow GL writes).
  std::array<float, 2> recolor_device(void* d_vertices, int n, int mode,
                                      const std::array<float, 3>& up);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sfm::viewer::gl
