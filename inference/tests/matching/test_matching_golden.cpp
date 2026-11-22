// Matching against OpenCV (golden files from tools/make_golden_p2.py, bunny 0012 / 0013).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "sfm/core/npy.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/features/orb.hpp"
#include "sfm/matching/bf_matcher.hpp"
#include "sfm/matching/verify.hpp"

#include "two_view.hpp"

namespace fs = std::filesystem;
using sfm::core::npy_read;

namespace {

const fs::path kSrc(SFM_SOURCE_DIR);
const fs::path kGolden = kSrc / "inference/tests/golden/p2";

// OpenCV's (n, 32) uint8 descriptors -> our 8 x u32 words (byte b goes to word b / 4,
// little-endian; the Hamming distance only counts differing bits, so any fixed byte order gives the
// same result).
std::vector<std::uint32_t> words(const sfm::core::NpyArray& a) {
  const auto bytes = a.to_vector<std::uint8_t>();
  std::vector<std::uint32_t> w(bytes.size() / 4);
  std::memcpy(w.data(), bytes.data(), bytes.size());
  return w;
}

// Symmetric epipolar distance (max over both images) in pixels.
double epipolar_px(const Eigen::Matrix3d& F, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  const Eigen::Vector3d l2 = F * a.homogeneous();
  const Eigen::Vector3d l1 = F.transpose() * b.homogeneous();
  const double d2 = std::fabs(l2.dot(b.homogeneous())) / l2.head<2>().norm();
  const double d1 = std::fabs(l1.dot(a.homogeneous())) / l1.head<2>().norm();
  return std::max(d1, d2);
}

}  // namespace

TEST(MatchingGolden, Knn2EqualsOpenCVBFMatcher) {
  const auto d12 = words(npy_read(kGolden / "orb_cv_0012_desc.npy"));
  const auto d13 = words(npy_read(kGolden / "orb_cv_0013_desc.npy"));
  const auto ref = npy_read(kGolden / "bf_knn_12_13.npy").to_vector<std::int32_t>();
  const auto k = sfm::matching::knn2(d12.data(), d12.size() / 8, d13.data(), d13.size() / 8);
  ASSERT_EQ(ref.size(), 4 * k.size());
  int index_ties = 0;
  for (std::size_t q = 0; q < k.size(); ++q) {
    ASSERT_EQ(k[q].best_distance, ref[4 * q + 1]) << q;
    ASSERT_EQ(k[q].second_distance, ref[4 * q + 3]) << q;
    // Indices must agree unless distances tie (then any order is a valid nearest-two).
    if (k[q].best_distance == k[q].second_distance) {
      ++index_ties;
      continue;
    }
    EXPECT_EQ(k[q].best, ref[4 * q]) << q;
    EXPECT_EQ(k[q].second, ref[4 * q + 2]) << q;
  }
  std::printf("%zu queries, %d with best == second distance\n", k.size(), index_ties);
}

TEST(MatchingGolden, OurOrbMatchesAsWellAsOpenCVs) {
  sfm::features::OrbParams orb;
  std::vector<sfm::features::Features> f;
  for (const char* id : {"0012", "0013"}) {
    const auto g = sfm::data::load_gray(kSrc / "data/bunny/images" / (std::string(id) + ".ppm"));
    const auto m =
        sfm::data::load_mask(kSrc / "data/bunny/silhouettes" / (std::string(id) + ".pgm"));
    f.push_back(sfm::features::detect_orb(g, &m, orb));
  }
  const auto P12 = sfm::data::read_projection(kSrc / "data/bunny/calib/0012.txt");
  const auto P13 = sfm::data::read_projection(kSrc / "data/bunny/calib/0013.txt");
  const Eigen::Matrix3d F = sfm::matching::test::fundamental_from_projections(P12, P13);

  const auto m = sfm::matching::match_ratio(f[0].desc.data(), f[0].size(), f[1].desc.data(),
                                            f[1].size(), 0.75, false);
  int good = 0;
  for (const auto& k : m) {
    const auto q = static_cast<std::size_t>(k.query);
    const auto t = static_cast<std::size_t>(k.train);
    good += epipolar_px(F, {f[0].xy[2 * q], f[0].xy[2 * q + 1]},
                        {f[1].xy[2 * t], f[1].xy[2 * t + 1]}) < 2.0;
  }
  const double share = static_cast<double>(good) / static_cast<double>(m.size());
  const double cv_share = npy_read(kGolden / "orb_cv_true_share_12_13.npy").to_vector<double>()[0];
  std::printf("ratio matches %zu, true share %.3f (OpenCV %.3f)\n", m.size(), share, cv_share);
  EXPECT_GT(share, cv_share - 0.05);

  // Verification keeps the true matches.
  sfm::matching::VerifyParams vp;
  sfm::matching::PairMatches pm;
  ASSERT_TRUE(sfm::matching::verify_pair(f[0], f[1], 12, 13, vp, pm));
  int in_good = 0;
  for (std::size_t k = 0; k < pm.size(); ++k) {
    const auto q = static_cast<std::size_t>(pm.idx[2 * k]);
    const auto t = static_cast<std::size_t>(pm.idx[2 * k + 1]);
    in_good += epipolar_px(F, {f[0].xy[2 * q], f[0].xy[2 * q + 1]},
                           {f[1].xy[2 * t], f[1].xy[2 * t + 1]}) < 2.0;
  }
  const double precision = static_cast<double>(in_good) / static_cast<double>(pm.size());
  std::printf("verified inliers %zu, precision %.3f\n", pm.size(), precision);
  EXPECT_GE(pm.size(), static_cast<std::size_t>(0.8 * good));
  EXPECT_GT(precision, 0.9);
}
