// Parity: SiftCuda::detect must equal detect_sift byte for byte (P7 gate).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include "sfm/data/image_io.hpp"
#include "sfm/features/sift.hpp"
#include "sfm/features/sift_cuda.hpp"

#include "image_util.hpp"
#include "test_env.hpp"

namespace fs = std::filesystem;
using sfm::features::Features;
using sfm::features::SiftCuda;
using sfm::features::SiftParams;

namespace {

// Counts the differing keypoints first, so a failure says how far apart the two are.
void expect_identical(const Features& cpu, const Features& gpu, const std::string& what) {
  EXPECT_EQ(gpu.words, sfm::features::kSiftWords) << what;
  ASSERT_EQ(cpu.size(), gpu.size()) << what;
  int diff = 0;
  for (std::size_t i = 0; i < cpu.size() && diff < 5; ++i) {
    const bool same = cpu.xy[2 * i] == gpu.xy[2 * i] && cpu.xy[2 * i + 1] == gpu.xy[2 * i + 1] &&
                      cpu.angle[i] == gpu.angle[i] && cpu.response[i] == gpu.response[i];
    if (!same) {
      ++diff;
      ADD_FAILURE() << what << " keypoint " << i << ": cpu (" << cpu.xy[2 * i] << ", "
                    << cpu.xy[2 * i + 1] << ", " << cpu.angle[i] << ") gpu (" << gpu.xy[2 * i]
                    << ", " << gpu.xy[2 * i + 1] << ", " << gpu.angle[i] << ")";
    }
  }
  EXPECT_EQ(cpu.octave, gpu.octave) << what;
  EXPECT_EQ(cpu.desc, gpu.desc) << what;
}

}  // namespace

TEST(SiftCuda, IdenticalToCpuOnSyntheticImages) {
  using sfm::features::test::make_image;
  const SiftParams p;
  const auto tex = make_image(160, 120, sfm::features::test::texture);
  SiftCuda gpu(p, 160, 120);
  expect_identical(sfm::features::detect_sift(tex, nullptr, p), gpu.detect(tex, nullptr),
                   "texture");
  const auto mask = make_image(160, 120, [](int x, int y) { return x + y < 150 ? 255 : 0; });
  expect_identical(sfm::features::detect_sift(tex, &mask, p), gpu.detect(tex, &mask), "masked");
  SiftParams q;  // other parameters: no upsampling, 4 scales, 1 orientation, few features
  q.first_octave = 0;
  q.octave_resolution = 4;
  q.max_orientations = 1;
  q.max_features = 40;
  SiftCuda gq(q, 160, 120);
  expect_identical(sfm::features::detect_sift(tex, nullptr, q), gq.detect(tex, nullptr), "params");
}

TEST(SiftCuda, IdenticalToCpuOnBunnyImages) {
  const fs::path root = fs::path(SFM_SOURCE_DIR) / "data" / "bunny";
  const SiftParams p;
  SiftCuda gpu(p, 1024, 768);
  std::size_t total = 0;
  for (int i = 0; i < 36; i += sfm::test::sanitizer_size(7, 36)) {  // 6 images, or image 0
    char id[16];
    std::snprintf(id, sizeof id, "%04d", i);
    const auto gray = sfm::data::load_gray(root / "images" / (std::string(id) + ".ppm"));
    const auto mask = sfm::data::load_mask(root / "silhouettes" / (std::string(id) + ".pgm"));
    const Features c = sfm::features::detect_sift(gray, &mask, p);
    const Features g = gpu.detect(gray, &mask);
    expect_identical(c, g, id);
    total += g.size();
    if (i == 0) {
      expect_identical(sfm::features::detect_sift(gray, nullptr, p), gpu.detect(gray, nullptr),
                       "no mask");
    }
  }
  EXPECT_GT(total, (sfm::test::under_sanitizer() ? 1u : 6u) * 300u);
}
