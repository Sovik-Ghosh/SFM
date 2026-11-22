#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "sfm/core/error.hpp"
#include "sfm/eval/rng.hpp"
#include "sfm/eval/umeyama.hpp"

using sfm::eval::Rng;
using sfm::eval::Similarity;
using sfm::eval::umeyama;

namespace {

Eigen::Matrix3d random_rotation(Rng& rng) {
  return Eigen::Quaterniond(rng.normal(), rng.normal(), rng.normal(), rng.normal())
      .normalized()
      .toRotationMatrix();
}

Eigen::Matrix3Xd random_points(Rng& rng, int n) {
  Eigen::Matrix3Xd p(3, n);
  for (int j = 0; j < n; ++j) p.col(j) << rng.normal(), rng.normal(), rng.normal();
  return p;
}

Eigen::Matrix3Xd transform(const Similarity& T, const Eigen::Matrix3Xd& p) {
  return ((T.s * T.R * p).colwise() + T.t).eval();
}

void expect_same(const Similarity& got, const Similarity& truth, double tol) {
  EXPECT_NEAR(got.s, truth.s, tol * truth.s);
  EXPECT_LT((got.R - truth.R).norm(), tol);
  EXPECT_LT((got.t - truth.t).norm(), tol * (1.0 + truth.t.norm()));
  EXPECT_NEAR(got.R.determinant(), 1.0, 1e-12);
}

}  // namespace

TEST(Umeyama, RecoversRandomSimilarities) {
  Rng rng(3);
  for (int trial = 0; trial < 100; ++trial) {
    SCOPED_TRACE(trial);
    const Similarity truth{
        rng.uniform(0.01, 100.0), random_rotation(rng),
        Eigen::Vector3d(rng.normal(0, 50), rng.normal(0, 50), rng.normal(0, 50))};
    const Eigen::Matrix3Xd src = random_points(rng, 50);
    expect_same(umeyama(src, transform(truth, src)), truth, 1e-9);
  }
}

TEST(Umeyama, ThreePointsAreEnough) {
  Rng rng(5);
  const Similarity truth{2.5, random_rotation(rng), Eigen::Vector3d(1, -2, 3)};
  const Eigen::Matrix3Xd src = random_points(rng, 3);
  expect_same(umeyama(src, transform(truth, src)), truth, 1e-9);
}

TEST(Umeyama, CoplanarPointsGiveARotationNotAReflection) {
  Rng rng(11);
  Eigen::Matrix3Xd src = random_points(rng, 20);
  src.row(2).setZero();  // all in the z = 0 plane: the cross-covariance has rank 2
  const Similarity truth{0.7, random_rotation(rng), Eigen::Vector3d(0, 4, 0)};
  expect_same(umeyama(src, transform(truth, src)), truth, 1e-9);
}

TEST(Umeyama, MirroredTargetStillGivesAProperRotation) {
  Rng rng(13);
  const Eigen::Matrix3Xd src = random_points(rng, 30);
  Eigen::Matrix3Xd dst = src;
  dst.row(0) *= -1.0;  // a reflection: no rotation fits exactly
  const Similarity got = umeyama(src, dst);
  EXPECT_NEAR(got.R.determinant(), 1.0, 1e-12);
  EXPECT_GT(got.s, 0.0);
}

TEST(Umeyama, NoiseGivesTheLeastSquaresFitNotAnExactOne) {
  Rng rng(17);
  const Similarity truth{3.0, random_rotation(rng), Eigen::Vector3d(5, 5, 5)};
  const Eigen::Matrix3Xd src = random_points(rng, 2000);
  Eigen::Matrix3Xd dst = transform(truth, src);
  for (Eigen::Index j = 0; j < dst.cols(); ++j) dst.col(j) += 0.01 * random_points(rng, 1).col(0);
  const Similarity got = umeyama(src, dst);
  // Fit residual must not exceed the residual of the true transform (least squares).
  EXPECT_LE((transform(got, src) - dst).squaredNorm(),
            (transform(truth, src) - dst).squaredNorm() * (1.0 + 1e-12));
  EXPECT_NEAR(got.s, 3.0, 1e-2);
}

TEST(Umeyama, DegenerateInputThrows) {
  Rng rng(19);
  const Eigen::Matrix3Xd two = random_points(rng, 2);
  EXPECT_THROW(umeyama(two, two), sfm::GeometryError);

  Eigen::Matrix3Xd line(3, 5);  // collinear: rotation about the line is free
  for (int j = 0; j < 5; ++j) line.col(j) = Eigen::Vector3d(1, 2, 3) * j;
  EXPECT_THROW(umeyama(line, line), sfm::GeometryError);

  const Eigen::Matrix3Xd same = Eigen::Matrix3Xd::Ones(3, 4);  // all points identical
  EXPECT_THROW(umeyama(same, same), sfm::GeometryError);

  EXPECT_THROW(umeyama(random_points(rng, 4), random_points(rng, 5)), sfm::GeometryError);
}
