#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "sfm/eval/rng.hpp"

using sfm::eval::Rng;

// The C++ standard fixes this value ([rand.predef]): the 10000th output of a default-constructed
// mt19937_64. If it fails, the engine is not the standard one and no seed below is reproducible.
TEST(Rng, EngineIsTheStandardMt19937_64) {
  std::mt19937_64 engine;
  engine.discard(9999);
  EXPECT_EQ(engine(), 9981545732273789042ull);
}

// Regression constants for seed 42. Recomputed independently in Python from the raw engine output
// ((bits >> 11) * 2^-53, Box-Muller with math.log/cos/sin). A change here changes every synthetic
// scene, so it must be deliberate.
TEST(Rng, FixedSeedGivesFixedValues) {
  Rng a(42);
  EXPECT_EQ(a.bits(), 13930160852258120406ull);
  Rng b(42);
  EXPECT_EQ(b.uniform(), 0.75515553295453897);  // exact: (bits >> 11) * 2^-53 is exact
  EXPECT_EQ(b.uniform(), 0.63903139385469743);
  EXPECT_EQ(b.uniform(), 0.7521452007480266);
  Rng c(42);
  const std::vector<double> normals{-1.0771745442782885, -1.2860634502166481, 1.0945198485006107,
                                    1.2616856516484893};
  for (double expected : normals) EXPECT_NEAR(c.normal(), expected, 1e-15);  // libm last-bit slack
}

TEST(Rng, UniformStaysInItsRange) {
  Rng rng(1);
  for (int i = 0; i < 100000; ++i) {
    const double u = rng.uniform();
    ASSERT_GE(u, 0.0);
    ASSERT_LT(u, 1.0);
    const double v = rng.uniform(-3.0, 5.0);
    ASSERT_GE(v, -3.0);
    ASSERT_LT(v, 5.0);
    ASSERT_LT(rng.index(7), 7u);
  }
}

TEST(Rng, NormalHasMeanZeroAndVarianceOne) {
  Rng rng(2);
  constexpr int kN = 200000;
  double sum = 0.0, sum_sq = 0.0;
  for (int i = 0; i < kN; ++i) {
    const double x = rng.normal();
    sum += x;
    sum_sq += x * x;
  }
  const double mean = sum / kN;
  EXPECT_NEAR(mean, 0.0, 0.01);                       // standard error 1/sqrt(N) ~ 0.0022
  EXPECT_NEAR(sum_sq / kN - mean * mean, 1.0, 0.01);  // standard error sqrt(2/N) ~ 0.0032
}

TEST(Rng, IndexCoversEveryValue) {
  Rng rng(4);
  std::vector<int> hits(10, 0);
  for (int i = 0; i < 10000; ++i) ++hits[rng.index(10)];
  for (int h : hits) EXPECT_NEAR(h, 1000, 150);  // binomial sd ~30
}
