#pragma once
// Stage contract of the feature stage: keypoints and descriptors of one image, as
// structure-of-arrays (the layout the GPU kernels use too). Two descriptor kinds, told apart by
// their width in 32-bit words: ORB = 256 bits (8 words, Hamming distance), SIFT = 128 bytes
// (32 words, byte k of the descriptor = bits 8 (k % 4) .. of word k / 4; L2 distance).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sfm::features {

inline constexpr int kDescriptorWords = 8;  // ORB: 256-bit descriptor = 8 x u32
inline constexpr int kSiftWords = 32;       // SIFT: 128 x u8 = 32 x u32

struct Features {
  std::vector<float> xy;             // (n, 2) keypoint position in level-0 pixels
  std::vector<float> angle;          // (n) orientation in degrees, [0, 360)
  std::vector<float> response;       // (n) ranking score (Harris response)
  std::vector<std::uint8_t> octave;  // (n) pyramid level the keypoint was found on
  std::vector<std::uint32_t> desc;   // (n, words): ORB bit k of word w = test pair 32 w + k
  int words = kDescriptorWords;      // descriptor width: kDescriptorWords (ORB) or kSiftWords

  std::size_t size() const { return angle.size(); }
  const std::uint32_t* descriptor(std::size_t i) const {
    return desc.data() + i * static_cast<std::size_t>(words);
  }
};

}  // namespace sfm::features
