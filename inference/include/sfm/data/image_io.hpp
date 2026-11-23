#pragma once
// Image loading for the dataset: RGB -> 8-bit gray, silhouette -> binary mask. Formats: whatever
// stb_image reads here: PPM/PGM (the bunny set), PNG and JPEG (baseline and progressive).
// load_gray and load_rgb apply a JPEG's EXIF orientation; load_mask does not (masks are drawn on
// the displayed image and saved without it). Pixels are row-major, width * height bytes.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sfm::data {

struct Image8 {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;  // row-major, width * height

  std::size_t size() const { return pixels.size(); }
};

// OpenCV's fixed-point BT.601 conversion, bit-exact with cv2.cvtColor(..., COLOR_RGB2GRAY):
// gray = (4899 R + 9617 G + 1868 B + 8192) >> 14.
inline std::uint8_t rgb_to_gray(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  return static_cast<std::uint8_t>((4899U * r + 9617U * g + 1868U * b + 8192U) >> 14);
}

// 8-bit RGB, interleaved, row-major: width * height * 3 bytes.
struct ImageRgb {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

// Any colour or gray image -> RGB (gray is replicated), as displayed: a JPEG's EXIF orientation is
// applied (a portrait phone photo stored landscape comes out portrait). Throws IoError.
ImageRgb load_rgb(const std::filesystem::path& path);

// Any colour or gray image -> gray (a 1-channel file is returned unchanged), with the EXIF
// orientation applied as in load_rgb. Throws IoError.
Image8 load_gray(const std::filesystem::path& path);

// Silhouette -> mask: 255 where the file is 0 (object), 0 elsewhere. The bunny silhouettes are not
// binary (19 distinct values), so this is an equality test, not a threshold. With object_white
// (dataset.mask_object "white"): 255 where the file is > 127, the usual convention. Throws IoError.
Image8 load_mask(const std::filesystem::path& path, bool object_white = false);

}  // namespace sfm::data
