// The interop kernels write sfm::viewer::Vertex through a mirror struct (interop.cu keeps Eigen out
// of nvcc); this pins the layout they assume.
#include <cstddef>

#include "sfm/viewer/scene.hpp"

namespace sfm::viewer {

static_assert(sizeof(Vertex) == 16, "interop.cu: 16-byte vertices");
static_assert(offsetof(Vertex, x) == 0 && offsetof(Vertex, y) == 4 && offsetof(Vertex, z) == 8,
              "interop.cu: position first");
static_assert(offsetof(Vertex, r) == 12 && offsetof(Vertex, g) == 13 && offsetof(Vertex, b) == 14 &&
                  offsetof(Vertex, a) == 15,
              "interop.cu: RGBA after the position");
static_assert(static_cast<int>(ColorMode::kGray) == 0 &&
                  static_cast<int>(ColorMode::kHeight) == 1 &&
                  static_cast<int>(ColorMode::kError) == 2 &&
                  static_cast<int>(ColorMode::kTrackLength) == 3,
              "colormap_hd.hpp: mode numbers");

}  // namespace sfm::viewer
