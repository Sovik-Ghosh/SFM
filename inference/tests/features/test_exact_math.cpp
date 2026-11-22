#include <gtest/gtest.h>

#include <cmath>

#include "sfm/features/exact_math.hpp"

namespace em = sfm::features::exact_math;

TEST(ExactMath, ExpWithinFewUlp) {
  double worst = 0.0;
  for (int k = 0; k <= 200000; ++k) {
    const float x = -40.0F + 40.0F * static_cast<float>(k) / 200000.0F;
    const double ref = std::exp(static_cast<double>(x));
    worst = std::max(worst, std::abs(em::exp(x) - ref) / ref);
  }
  EXPECT_LT(worst, 4e-7);  // ~3 ulp
  EXPECT_EQ(em::exp(0.0F), 1.0F);
  EXPECT_EQ(em::exp(-100.0F), 0.0F);
}

TEST(ExactMath, Atan2CoversAllQuadrantsIn0To2Pi) {
  double worst = 0.0;
  for (int k = 0; k < 100000; ++k) {
    const double t = 2.0 * M_PI * k / 100000.0;
    const auto x = static_cast<float>(3.0 * std::cos(t));
    const auto y = static_cast<float>(3.0 * std::sin(t));
    double ref = std::atan2(static_cast<double>(y), static_cast<double>(x));
    if (ref < 0.0) ref += 2.0 * M_PI;
    double d = std::abs(em::atan2_2pi(y, x) - ref);
    d = std::min(d, 2.0 * M_PI - d);  // wrap at 0 / 2 pi
    worst = std::max(worst, d);
  }
  EXPECT_LT(worst, 1e-6);
  EXPECT_EQ(em::atan2_2pi(0.0F, 0.0F), 0.0F);
  EXPECT_EQ(em::atan2_2pi(0.0F, 1.0F), 0.0F);
  EXPECT_NEAR(em::atan2_2pi(-1.0F, 0.0F), 1.5 * M_PI, 1e-6);
}

TEST(ExactMath, SinCosOverSeveralTurns) {
  double worst = 0.0;
  for (int k = 0; k <= 100000; ++k) {
    const float x = -7.0F + 21.0F * static_cast<float>(k) / 100000.0F;
    float s = 0.0F;
    float c = 0.0F;
    em::sincos(x, s, c);
    worst = std::max({worst, std::abs(s - std::sin(static_cast<double>(x))),
                      std::abs(c - std::cos(static_cast<double>(x)))});
  }
  EXPECT_LT(worst, 3e-7);
}
