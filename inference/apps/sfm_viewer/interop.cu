#include "interop.hpp"

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
// clang-format off: the GL header must come before the CUDA-GL interop header.
#include <cuda_gl_interop.h>
// clang-format on

#include <cmath>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sfm/core/cuda_check.h"
#include "sfm/viewer/colormap_hd.hpp"

namespace sfm::viewer::gl {

namespace {

// sfm::viewer::Vertex (scene.hpp, which brings Eigen and is kept out of nvcc): 3 floats then RGBA
// bytes. The layout is checked against Vertex in interop_layout.cpp.
struct GpuVertex {
  float x;
  float y;
  float z;
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};
static_assert(sizeof(GpuVertex) == 16, "16-byte vertices");

__global__ void values_kernel(const GpuVertex* v, int n, int mode, float ux, float uy, float uz,
                              const std::uint8_t* gray, const float* error, const int* track,
                              float* out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = hd::point_value(mode, v[i].x, v[i].y, v[i].z, ux, uy, uz, gray[i], error[i], track[i]);
}

__global__ void colors_kernel(GpuVertex* v, int n, int mode, const float* value,
                              const std::uint8_t* gray, float lo, float hi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  std::uint8_t c[3];
  if (mode == 0) {
    c[0] = c[1] = c[2] = gray[i];
  } else {
    hd::viridis(hd::normalise(value[i], lo, hi), c);
  }
  v[i].r = c[0];
  v[i].g = c[1];
  v[i].b = c[2];
}

// robust_range's quantile index (colormap.cpp), on the host.
std::size_t quantile_index(float q, std::size_t n) {
  const auto k = static_cast<std::size_t>(std::lround(q * static_cast<float>(n - 1)));
  return k < n ? k : n - 1;
}

}  // namespace

struct PointColorsCuda::Impl {
  cudaGraphicsResource* resource = nullptr;
  int count = 0;
  thrust::device_vector<std::uint8_t> gray;
  thrust::device_vector<float> error;
  thrust::device_vector<int> track;
  thrust::device_vector<float> value;
  thrust::device_vector<float> sorted;

  void detach() {
    if (resource != nullptr) cudaGraphicsUnregisterResource(resource);  // no throw in cleanup
    resource = nullptr;
  }
};

bool PointColorsCuda::available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    cudaGetLastError();  // clear the sticky error of a missing driver
    return false;
  }
  unsigned int found = 0;
  int devices[4] = {};
  const cudaError_t e = cudaGLGetDevices(&found, devices, 4, cudaGLDeviceListAll);
  if (e != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return found > 0;
}

PointColorsCuda::PointColorsCuda() : impl_(std::make_unique<Impl>()) {}

PointColorsCuda::~PointColorsCuda() { impl_->detach(); }

void PointColorsCuda::attach(unsigned int vbo, int count) {
  impl_->detach();
  impl_->count = count;
  if (count <= 0) return;
  CHECK(cudaGraphicsGLRegisterBuffer(&impl_->resource, vbo, cudaGraphicsRegisterFlagsNone));
}

void PointColorsCuda::set_attributes(const std::vector<std::uint8_t>& gray,
                                     const std::vector<float>& error_px,
                                     const std::vector<int>& track_length) {
  impl_->gray.assign(gray.begin(), gray.end());
  impl_->error.assign(error_px.begin(), error_px.end());
  impl_->track.assign(track_length.begin(), track_length.end());
}

std::array<float, 2> PointColorsCuda::recolor(int mode, const std::array<float, 3>& up) {
  Impl& m = *impl_;
  if (m.count <= 0 || m.resource == nullptr) return {0.0F, 1.0F};
  CHECK(cudaGraphicsMapResources(1, &m.resource));
  void* v = nullptr;
  std::size_t bytes = 0;
  std::array<float, 2> range = {0.0F, 1.0F};
  try {
    CHECK(cudaGraphicsResourceGetMappedPointer(&v, &bytes, m.resource));
    range = recolor_device(v, m.count, mode, up);
  } catch (...) {
    cudaGraphicsUnmapResources(1, &m.resource);
    throw;
  }
  CHECK(cudaGraphicsUnmapResources(1, &m.resource));  // OpenGL sees the colours after this
  return range;
}

std::array<float, 2> PointColorsCuda::recolor_device(void* d_vertices, int n, int mode,
                                                     const std::array<float, 3>& up) {
  Impl& m = *impl_;
  if (n <= 0) return {0.0F, 1.0F};
  if (static_cast<int>(m.gray.size()) != n || static_cast<int>(m.error.size()) != n ||
      static_cast<int>(m.track.size()) != n) {
    throw core::CudaError(cudaErrorInvalidValue, "point attributes do not match the buffer",
                          __FILE__, __LINE__);
  }
  auto* v = static_cast<GpuVertex*>(d_vertices);
  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  m.value.resize(static_cast<std::size_t>(n));
  values_kernel<<<grid, kBlock>>>(
      v, n, mode, up[0], up[1], up[2], thrust::raw_pointer_cast(m.gray.data()),
      thrust::raw_pointer_cast(m.error.data()), thrust::raw_pointer_cast(m.track.data()),
      thrust::raw_pointer_cast(m.value.data()));
  CHECK(cudaGetLastError());
  std::array<float, 2> range = {0.0F, 1.0F};
  if (mode != 0) {  // not gray: robust_range, the 5 % and 95 % quantiles
    m.sorted = m.value;
    thrust::sort(m.sorted.begin(), m.sorted.end());
    const auto un = static_cast<std::size_t>(n);
    const float lo = m.sorted[quantile_index(0.05F, un)];
    float hi = m.sorted[quantile_index(0.95F, un)];
    if (!(hi > lo)) hi = lo + 1.0F;
    range = {lo, hi};
  }
  colors_kernel<<<grid, kBlock>>>(v, n, mode, thrust::raw_pointer_cast(m.value.data()),
                                  thrust::raw_pointer_cast(m.gray.data()), range[0], range[1]);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  return range;
}

}  // namespace sfm::viewer::gl
