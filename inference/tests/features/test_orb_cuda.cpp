// Parity: OrbCuda::detect must equal detect_orb bit for bit (P3 gate).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include "sfm/data/image_io.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/features/orb.hpp"
#include "sfm/features/orb_cuda.hpp"

#include "image_util.hpp"
#include "test_env.hpp"

namespace fs = std::filesystem;
using sfm::features::Features;
using sfm::features::OrbCuda;
using sfm::features::OrbParams;

namespace {

void expect_identical(const Features& cpu, const Features& gpu, const std::string& what) {
  ASSERT_EQ(cpu.size(), gpu.size()) << what;
  EXPECT_EQ(cpu.xy, gpu.xy) << what;
  EXPECT_EQ(cpu.octave, gpu.octave) << what;
  EXPECT_EQ(cpu.response, gpu.response) << what;
  EXPECT_EQ(cpu.angle, gpu.angle) << what;
  EXPECT_EQ(cpu.desc, gpu.desc) << what;
}

}  // namespace

TEST(OrbCuda, IdenticalToCpuOnAllBunnyImages) {
  const fs::path root = fs::path(SFM_SOURCE_DIR) / "data" / "bunny";
  const OrbParams p;
  OrbCuda gpu(p, 1024, 768);
  std::size_t total = 0;
  const int n = sfm::test::sanitizer_size(36, 2);
  for (int i = 0; i < n; ++i) {
    char id[16];
    std::snprintf(id, sizeof id, "%04d", i);
    const auto gray = sfm::data::load_gray(root / "images" / (std::string(id) + ".ppm"));
    const auto mask = sfm::data::load_mask(root / "silhouettes" / (std::string(id) + ".pgm"));
    const Features c = sfm::features::detect_orb(gray, &mask, p);
    const Features g = gpu.detect(gray, &mask);
    expect_identical(c, g, id);
    total += g.size();
  }
  EXPECT_GT(total, static_cast<std::size_t>(n) * 1000u);
}

TEST(OrbCuda, IdenticalToCpuOnRandomImagesWithoutMask) {
  // Noise gives many corners and many score / Harris ties: exercises the tie rules.
  OrbParams p;
  p.max_features = 3000;
  p.levels = 5;
  p.scale = 1.3;
  p.fast_threshold = 20;
  OrbCuda gpu(p, 640, 480);
  for (std::uint64_t seed : {1, 2, 3}) {
    sfm::eval::Rng rng(seed);
    const auto img = sfm::features::test::make_image(
        640, 480, [&](int, int) { return static_cast<int>(rng.index(256)); });
    expect_identical(sfm::features::detect_orb(img, nullptr, p), gpu.detect(img, nullptr),
                     "seed " + std::to_string(seed));
  }
}

TEST(OrbCuda, RejectsOtherSizes) {
  OrbCuda gpu(OrbParams{}, 320, 240);
  const auto img = sfm::features::test::make_image(321, 240, [](int, int) { return 0; });
  EXPECT_THROW(gpu.detect(img, nullptr), std::invalid_argument);
  EXPECT_THROW(OrbCuda(OrbParams{}, 0, 10), std::invalid_argument);
}
