// RGB pixels -> PNG (stb_image_write; its implementation lives in this file only).
#include <cstdint>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "sfm/core/error.hpp"

#include "app.hpp"
#include "stb_image_write.h"

namespace sfm::viewer {

void save_png(const std::filesystem::path& file, int width, int height,
              const std::vector<std::uint8_t>& rgb) {
  const int stride = 3 * width;
  if (stbi_write_png(file.string().c_str(), width, height, 3, rgb.data(), stride) == 0) {
    throw IoError("screenshot: cannot write " + file.string());
  }
}

}  // namespace sfm::viewer
