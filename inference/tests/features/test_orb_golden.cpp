// Statistical comparison of our ORB with OpenCV's on bunny 0012 (golden files from
// tools/make_golden_p2.py). Bit-exactness is not a goal: the test pattern, pyramid resampling and
// level-0 mapping differ on purpose (see orb.hpp). The matching quality comparison lives in the
// matching tests.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "sfm/core/npy.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/features/orb.hpp"

namespace fs = std::filesystem;

TEST(OrbGolden, KeypointsCoincideWithOpenCV) {
  const fs::path src(SFM_SOURCE_DIR);
  const fs::path golden = src / "inference/tests/golden/p2";
  auto cv_xy = sfm::core::npy_read(golden / "orb_cv_0012_xy.npy").to_vector<float>();
  const auto cv_octave =
      sfm::core::npy_read(golden / "orb_cv_0012_octave.npy").to_vector<std::int32_t>();
  ASSERT_EQ(cv_xy.size(), 2 * cv_octave.size());
  // OpenCV maps a level-l pixel x to x * s^l; the pixel-centre convention we use gives
  // (x + 0.5) s^l - 0.5. Shift OpenCV's points by 0.5 (s^l - 1) so both use the same convention.
  for (std::size_t j = 0; j < cv_octave.size(); ++j) {
    const auto shift = static_cast<float>(0.5 * (std::pow(1.2, cv_octave[j]) - 1.0));
    cv_xy[2 * j] += shift;
    cv_xy[2 * j + 1] += shift;
  }
  const auto gray = sfm::data::load_gray(src / "data/bunny/images/0012.ppm");
  const auto mask = sfm::data::load_mask(src / "data/bunny/silhouettes/0012.pgm");
  const auto f = sfm::features::detect_orb(gray, &mask, sfm::features::OrbParams{});

  // Per level: how many of our keypoints have an OpenCV keypoint of the same level within 1 px.
  std::vector<std::size_t> n(8, 0);
  std::vector<std::size_t> hit(8, 0);
  for (std::size_t i = 0; i < f.size(); ++i) {
    const std::size_t l = f.octave[i];
    ++n[l];
    for (std::size_t j = 0; j < cv_octave.size(); ++j) {
      if (static_cast<std::size_t>(cv_octave[j]) == l &&
          std::hypot(f.xy[2 * i] - cv_xy[2 * j], f.xy[2 * i + 1] - cv_xy[2 * j + 1]) <= 1.0) {
        ++hit[l];
        break;
      }
    }
  }
  std::size_t total = 0;
  for (std::size_t l = 0; l < 8; ++l) {
    std::printf("level %zu: %zu keypoints, %.3f within 1 px of OpenCV\n", l, n[l],
                static_cast<double>(hit[l]) / static_cast<double>(n[l]));
    total += hit[l];
  }
  const double share = static_cast<double>(total) / static_cast<double>(f.size());
  std::printf("all: ours %zu, OpenCV %zu, %.3f within 1 px\n", f.size(), cv_octave.size(), share);
  // Measured 2026-11-19 (OpenCV 4.14): 0.832 overall; level 0: 0.899, level 7: 0.689. The
  // finest level uses the same pixels and the same FAST test, so it agrees best; higher levels
  // differ by resampling and by Harris-ranking ties.
  EXPECT_GE(share, 0.80);
  EXPECT_GE(static_cast<double>(hit[0]) / static_cast<double>(n[0]), 0.85);
}
