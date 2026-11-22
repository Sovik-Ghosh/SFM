#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sfm/data/calib.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/krt.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/synthetic.hpp"
#include "sfm/features/orb.hpp"
#include "sfm/matching/matching_cuda.hpp"

#include "test_env.hpp"
#include "two_view.hpp"

namespace fs = std::filesystem;
using namespace sfm::matching;  // NOLINT(google-build-using-namespace)

namespace {

const std::vector<sfm::features::Features>& bunny_features() {
  static const std::vector<sfm::features::Features> kF = [] {
    const fs::path root = fs::path(SFM_SOURCE_DIR) / "data" / "bunny";
    std::vector<sfm::features::Features> f;
    for (int i = 0; i < 36; ++i) {
      char id[16];
      std::snprintf(id, sizeof id, "%04d", i);
      const auto g = sfm::data::load_gray(root / "images" / (std::string(id) + ".ppm"));
      const auto m = sfm::data::load_mask(root / "silhouettes" / (std::string(id) + ".pgm"));
      f.push_back(sfm::features::detect_orb(g, &m, sfm::features::OrbParams{}));
    }
    return f;
  }();
  return kF;
}

struct TwoViews {
  Points2 x1;
  Points2 x2;
  Eigen::Matrix3d F;
};

TwoViews two_views(double noise, std::uint64_t seed) {
  sfm::eval::SyntheticConfig c;
  c.cameras = 2;
  c.points = 200;
  c.box_half = 20.0;
  c.noise_px = noise;
  c.seed = seed;
  const auto s = sfm::eval::make_synthetic_scene(c);
  TwoViews v;
  for (std::size_t p = 0; p < 200; ++p) {
    v.x1.push_back(s.observations[p].uv);
    v.x2.push_back(s.observations[200 + p].uv);
  }
  Eigen::Matrix<double, 3, 4> P1;
  Eigen::Matrix<double, 3, 4> P2;
  P1 << s.K * s.poses[0].R, s.K * s.poses[0].t;
  P2 << s.K * s.poses[1].R, s.K * s.poses[1].t;
  v.F = sfm::matching::test::fundamental_from_projections(P1, P2);
  return v;
}

}  // namespace

TEST(Knn2Cuda, EqualsTheCpuOnAllBunnyPairs) {
  const auto& f = bunny_features();
  const int n = sfm::test::sanitizer_size(36, 4);
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i != j) pairs.emplace_back(i, j);  // both directions (mutual uses the reverse)
    }
  }
  const auto gpu = knn2_cuda(f, pairs);
  ASSERT_EQ(gpu.size(), pairs.size());
  std::size_t queries = 0;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto& a = f[static_cast<std::size_t>(pairs[k].first)];
    const auto& b = f[static_cast<std::size_t>(pairs[k].second)];
    const auto cpu = knn2(a.desc.data(), a.size(), b.desc.data(), b.size());
    ASSERT_EQ(gpu[k].size(), cpu.size());
    for (std::size_t q = 0; q < cpu.size(); ++q) {
      ASSERT_EQ(gpu[k][q].best, cpu[q].best) << k << " " << q;
      ASSERT_EQ(gpu[k][q].best_distance, cpu[q].best_distance);
      ASSERT_EQ(gpu[k][q].second, cpu[q].second);
      ASSERT_EQ(gpu[k][q].second_distance, cpu[q].second_distance);
    }
    queries += cpu.size();
  }
  EXPECT_GT(queries, sfm::test::under_sanitizer() ? 1000u : 1000000u);
  // Empty and single-descriptor images.
  std::vector<sfm::features::Features> small(2);
  small[1].desc.assign(8, 0xFFU);
  small[1].angle.assign(1, 0.0F);
  const auto r = knn2_cuda(small, {{0, 1}, {1, 0}});
  EXPECT_TRUE(r[0].empty());
  ASSERT_EQ(r[1].size(), 1u);
  EXPECT_EQ(r[1][0].best, -1);
}

TEST(Knn2Cuda, SiftL2EqualsTheCpuIncludingTiesAndEmptyImages) {
  // Random 128-byte descriptors; image sizes around the 64-descriptor tile, an empty image, one
  // descriptor, and copies (ties go to the lower index on both sides).
  sfm::eval::Rng rng(7);
  std::vector<sfm::features::Features> f;
  for (const int n : {0, 1, 63, 64, 65, 200, 1000}) {
    sfm::features::Features x;
    x.words = sfm::features::kSiftWords;
    x.angle.assign(static_cast<std::size_t>(n), 0.0F);
    for (int k = 0; k < 32 * n; ++k) x.desc.push_back(static_cast<std::uint32_t>(rng.bits()));
    if (n >= 64) {  // descriptor 5 repeated at 40: equal distances to every query
      constexpr std::ptrdiff_t kWords = 32;
      std::copy_n(x.desc.begin() + 5 * kWords, kWords, x.desc.begin() + 40 * kWords);
    }
    f.push_back(std::move(x));
  }
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) pairs.emplace_back(i, j);
  }
  const auto gpu = knn2_cuda(f, pairs);
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto& a = f[static_cast<std::size_t>(pairs[k].first)];
    const auto& b = f[static_cast<std::size_t>(pairs[k].second)];
    const auto cpu =
        knn2(a.desc.data(), a.size(), b.desc.data(), b.size(), sfm::features::kSiftWords);
    ASSERT_EQ(gpu[k].size(), cpu.size());
    for (std::size_t q = 0; q < cpu.size(); ++q) {
      ASSERT_EQ(gpu[k][q].best, cpu[q].best) << k << " " << q;
      ASSERT_EQ(gpu[k][q].best_distance, cpu[q].best_distance);
      ASSERT_EQ(gpu[k][q].second, cpu[q].second);
      ASSERT_EQ(gpu[k][q].second_distance, cpu[q].second_distance);
    }
  }
}

TEST(RansacCuda, NoiseFreeEqualsTheCpuEightPointOnAllPoints) {
  const auto v = two_views(0.0, 3);
  RansacParams p;
  const auto r = ransac_fundamental_cuda({v.x1}, {v.x2}, {7}, p);
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].inliers, 200);
  EXPECT_EQ(r[0].iterations, p.max_iters);
  // The host refit runs eight_point on all 200 inliers: exactly the CPU's F.
  EXPECT_EQ(r[0].F, eight_point(v.x1, v.x2));
  EXPECT_LT(sfm::matching::test::f_distance(r[0].F, v.F), 1e-9);
}

TEST(RansacCuda, FindsInliersAmongOutliersDeterministically) {
  std::vector<Points2> x1;
  std::vector<Points2> x2;
  std::vector<std::vector<bool>> outlier;
  std::vector<std::uint64_t> seeds;
  for (std::uint64_t s = 1; s <= 6; ++s) {
    auto v = two_views(0.3, s);
    sfm::eval::Rng rng(100 + s);
    std::vector<bool> out(v.x1.size(), false);
    for (std::size_t k = 0; k < v.x1.size(); ++k) {
      if (k % 5 < 2) {
        v.x2[k] = {rng.uniform(0, 1024), rng.uniform(0, 768)};
        out[k] = true;
      }
    }
    x1.push_back(v.x1);
    x2.push_back(v.x2);
    outlier.push_back(out);
    seeds.push_back(s);
  }
  x1.emplace_back(x1[0].begin(), x1[0].begin() + 7);  // too few points: no model
  x2.emplace_back(x2[0].begin(), x2[0].begin() + 7);
  seeds.push_back(99);
  RansacParams p;
  p.threshold_px = 2.0;
  const auto r = ransac_fundamental_cuda(x1, x2, seeds, p);
  ASSERT_EQ(r.size(), 7u);
  for (std::size_t k = 0; k < 6; ++k) {
    int true_in = 0;
    int false_in = 0;
    for (std::size_t m = 0; m < x1[k].size(); ++m) {
      if (r[k].inlier[m] != 0) (outlier[k][m] ? false_in : true_in)++;
    }
    EXPECT_EQ(true_in, 120) << k;
    EXPECT_LE(false_in, 3) << k;
  }
  EXPECT_EQ(r[6].inliers, 0);
  EXPECT_EQ(r[6].iterations, 0);
  const auto again = ransac_fundamental_cuda(x1, x2, seeds, p);
  for (std::size_t k = 0; k < r.size(); ++k) {
    EXPECT_EQ(again[k].F, r[k].F);
    EXPECT_EQ(again[k].inlier, r[k].inlier);
  }
}

TEST(VerifyCuda, BunnyPairsAgreeWithTheCpu) {
  const auto& all = bunny_features();
  const std::vector<sfm::features::Features> f(
      all.begin(), all.begin() + sfm::test::sanitizer_size(static_cast<int>(all.size()), 6));
  VerifyParams p;
  p.seed = 42;
  const auto cpu = verify_all_pairs(f, p);
  const auto gpu = verify_all_pairs_cuda(f, p);
  std::set<std::pair<int, int>> a;
  std::set<std::pair<int, int>> b;
  for (const auto& m : cpu) a.insert({m.i, m.j});
  for (const auto& m : gpu) b.insert({m.i, m.j});
  std::size_t common = 0;
  for (const auto& k : a) common += b.count(k);
  std::printf("verified pairs: cpu %zu, gpu %zu, common %zu\n", a.size(), b.size(), common);
  // RANSAC samples differently, so pairs near min_inliers can flip; most must agree.
  if (!sfm::test::under_sanitizer()) {  // too few pairs there for a 90 % rate
    EXPECT_GE(common, a.size() * 9 / 10);
    EXPECT_GE(common, b.size() * 9 / 10);
  }
  const auto again = verify_all_pairs_cuda(f, p);
  ASSERT_EQ(again.size(), gpu.size());
  for (std::size_t k = 0; k < gpu.size(); ++k) EXPECT_EQ(again[k].idx, gpu[k].idx);
}

// The top-2 runs in chunks of pairs (P9a: bounded host / device memory on ~1000-image scenes);
// many tiny chunks give exactly the result of one.
TEST(VerifyCuda, ChunkedTop2GivesTheSameResult) {
  const auto& f = bunny_features();
  VerifyParams p;
  p.mutual = true;
  // The pairs of one image (35) cross many chunk boundaries; all 630 pairs made this the slowest
  // test under compute-sanitizer (racecheck > 15 min). Image 3 has verified pairs both ways round
  // the turntable (image 0 has none with ORB).
  std::vector<std::pair<int, int>> pairs;
  for (const auto& ij : select_pairs(static_cast<int>(f.size()), "exhaustive", 0, 0)) {
    if (ij.first == 3 || ij.second == 3) pairs.push_back(ij);
  }
  ASSERT_EQ(pairs.size(), 35U);
  if (sfm::test::under_sanitizer()) pairs.resize(8);  // still several chunks
  const auto one = verify_pairs_cuda(f, pairs, p);
  const auto many = verify_pairs_cuda(f, pairs, p, 3000);  // about one pair per chunk
  ASSERT_GE(one.size(), sfm::test::under_sanitizer() ? 1U : 5U);
  ASSERT_EQ(one.size(), many.size());
  for (std::size_t k = 0; k < one.size(); ++k) {
    EXPECT_EQ(one[k].i, many[k].i);
    EXPECT_EQ(one[k].j, many[k].j);
    EXPECT_EQ(one[k].idx, many[k].idx);
    EXPECT_EQ(one[k].F, many[k].F);
  }
}

// With the essential model the E-RANSAC of a chunk runs on worker threads while the next chunk's
// top-2 runs on the GPU (P9c): many chunks still give exactly the result of one.
TEST(VerifyCuda, ChunkedEssentialOverlapGivesTheSameResult) {
  const auto& f = bunny_features();
  const fs::path calib = fs::path(SFM_SOURCE_DIR) / "data" / "bunny" / "calib";
  std::vector<Eigen::Matrix3d> K;
  for (int i = 0; i < 36; ++i) {
    char id[16];
    std::snprintf(id, sizeof id, "%04d", i);
    K.push_back(sfm::data::decompose_projection(
                    sfm::data::read_projection(calib / (std::string(id) + ".txt")))
                    .K);
  }
  VerifyParams p;
  p.mutual = true;
  p.model = "essential";
  p.K = &K;
  std::vector<std::pair<int, int>> pairs;
  for (const auto& ij : select_pairs(static_cast<int>(f.size()), "exhaustive", 0, 0)) {
    if (ij.first == 3 || ij.second == 3) pairs.push_back(ij);
  }
  if (sfm::test::under_sanitizer()) pairs.resize(8);
  const auto one = verify_pairs_cuda(f, pairs, p);
  const auto many = verify_pairs_cuda(f, pairs, p, 3000);  // about one pair per chunk
  ASSERT_GE(one.size(), sfm::test::under_sanitizer() ? 1U : 5U);
  ASSERT_EQ(one.size(), many.size());
  for (std::size_t k = 0; k < one.size(); ++k) {
    EXPECT_EQ(one[k].i, many[k].i);
    EXPECT_EQ(one[k].j, many[k].j);
    EXPECT_EQ(one[k].idx, many[k].idx);
    EXPECT_EQ(one[k].F, many[k].F);
  }
}
