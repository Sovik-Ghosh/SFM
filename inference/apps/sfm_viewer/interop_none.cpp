// CPU-only build (SFM_WITH_CUDA=OFF): no CUDA-OpenGL interop. available() is false, so the viewer
// never constructs PointColorsCuda and colours points on the CPU.
#include <stdexcept>

#include "interop.hpp"

namespace sfm::viewer::gl {

struct PointColorsCuda::Impl {};

bool PointColorsCuda::available() { return false; }
PointColorsCuda::PointColorsCuda() {
  throw std::runtime_error("PointColorsCuda: this build has no CUDA support");
}
PointColorsCuda::~PointColorsCuda() = default;
void PointColorsCuda::attach(unsigned int /*vbo*/, int /*count*/) {}
void PointColorsCuda::set_attributes(const std::vector<std::uint8_t>& /*gray*/,
                                     const std::vector<float>& /*error_px*/,
                                     const std::vector<int>& /*track_length*/) {}
std::array<float, 2> PointColorsCuda::recolor(int /*mode*/, const std::array<float, 3>& /*up*/) {
  return {0.0F, 1.0F};
}
std::array<float, 2> PointColorsCuda::recolor_device(void* /*d_vertices*/, int /*n*/, int /*mode*/,
                                                     const std::array<float, 3>& /*up*/) {
  return {0.0F, 1.0F};
}

}  // namespace sfm::viewer::gl
