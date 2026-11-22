#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"
#include "sfm/data/krt.hpp"

using sfm::data::compose_projection;
using sfm::data::decompose_projection;
using sfm::data::Krt;
using sfm::data::Mat34;

namespace {

Krt random_camera(std::mt19937_64& gen) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  Krt c;
  c.K << 1200 + 200 * u(gen), 2 * u(gen), 500 + 50 * u(gen),  //
      0, 1200 + 200 * u(gen), 400 + 50 * u(gen),              //
      0, 0, 1;
  c.R = Eigen::Quaterniond(u(gen), u(gen), u(gen), u(gen)).normalized().toRotationMatrix();
  c.t << 10 * u(gen), 10 * u(gen), 60 + 10 * u(gen);
  return c;
}

void expect_valid(const Krt& c) {
  EXPECT_NEAR(c.K(2, 2), 1.0, 1e-12);
  EXPECT_GT(c.K(0, 0), 0.0);
  EXPECT_GT(c.K(1, 1), 0.0);
  EXPECT_EQ(c.K(1, 0), 0.0);  // exactly upper-triangular
  EXPECT_EQ(c.K(2, 0), 0.0);
  EXPECT_EQ(c.K(2, 1), 0.0);
  EXPECT_LT((c.R * c.R.transpose() - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(c.R.determinant(), 1.0, 1e-12);
}

}  // namespace

TEST(Krt, ComposeIsKTimesRt) {
  std::mt19937_64 gen(1);
  const Krt c = random_camera(gen);
  const Mat34 P = compose_projection(c);
  EXPECT_LT((P.leftCols<3>() - c.K * c.R).norm(), 1e-9);
  EXPECT_LT((P.col(3) - c.K * c.t).norm(), 1e-9);
}

TEST(Krt, RecoversRandomCamerasAtAnyScale) {
  std::mt19937_64 gen(42);
  for (int i = 0; i < 200; ++i) {
    const Krt truth = random_camera(gen);
    for (double scale : {1.0, 1e-3, 37.5, -1.0, -0.02}) {  // negative: sign of lambda flips
      SCOPED_TRACE(testing::Message() << "camera " << i << " scale " << scale);
      const Krt got = decompose_projection(scale * compose_projection(truth));
      expect_valid(got);
      EXPECT_LT((got.K - truth.K).norm() / truth.K.norm(), 1e-9);
      EXPECT_LT((got.R - truth.R).norm(), 1e-9);
      EXPECT_LT((got.t - truth.t).norm() / truth.t.norm(), 1e-9);
    }
  }
}

TEST(Krt, IdentityCamera) {
  Mat34 P = Mat34::Zero();
  P.leftCols<3>().setIdentity();
  const Krt c = decompose_projection(P);
  EXPECT_LT((c.K - Eigen::Matrix3d::Identity()).norm(), 1e-15);
  EXPECT_LT((c.R - Eigen::Matrix3d::Identity()).norm(), 1e-15);
  EXPECT_LT(c.t.norm(), 1e-15);
}

TEST(Krt, CentreIsNullSpaceOfP) {
  std::mt19937_64 gen(7);
  const Krt c = decompose_projection(-3.0 * compose_projection(random_camera(gen)));
  const Mat34 P = compose_projection(c);
  EXPECT_LT((P * c.centre().homogeneous()).norm(), 1e-9);  // P C = 0: the centre projects nowhere
}

TEST(Krt, SingularLeftBlockThrows) {
  Mat34 P = Mat34::Zero();
  P(0, 0) = 1;
  P(1, 1) = 1;  // third row zero: rank 2
  EXPECT_THROW(decompose_projection(P), sfm::IoError);
  EXPECT_THROW(decompose_projection(Mat34::Zero()), sfm::IoError);
}
