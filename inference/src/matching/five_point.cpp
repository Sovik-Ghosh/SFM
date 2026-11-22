// Five-point relative pose, action-matrix form (Stewénius, Engels, Nistér, "Recent developments on
// direct relative orientation", ISPRS 2006):
//   1. each correspondence gives one row q2^T E q1 = 0 of a 5x9 system; E = x X + y Y + z Z + W
//      with X, Y, Z, W spanning its null space;
//   2. the ten cubic constraints det(E) = 0 and 2 E E^T E - tr(E E^T) E = 0 in (x, y, z), written
//      over the 20 monomials of degree <= 3;
//   3. Gauss-Jordan elimination of the 10 cubic monomials leaves 10 = number of solutions;
//      multiplication by x on the basis [x^2 xy xz y^2 yz z^2 x y z 1] is a 10x10 action matrix
//      whose eigenvectors are that basis evaluated at the solutions.
#include "sfm/matching/five_point.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include "sfm/eval/rng.hpp"

namespace sfm::matching {

namespace {

// Polynomial in x, y, z of degree <= 3 over the 20 monomials, in this order:
//   0 x^3 1 x^2y 2 x^2z 3 xy^2 4 xyz 5 xz^2 6 y^3 7 y^2z 8 yz^2 9 z^3
//   10 x^2 11 xy 12 xz 13 y^2 14 yz 15 z^2 16 x 17 y 18 z 19 1
using Poly = std::array<double, 20>;

constexpr int kExpo[20][3] = {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1},
                              {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3},
                              {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1},
                              {0, 0, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};

// Index of x^a y^b z^c (a + b + c <= 3).
constexpr int mono(int a, int b, int c) {
  for (int k = 0; k < 20; ++k) {
    if (kExpo[k][0] == a && kExpo[k][1] == b && kExpo[k][2] == c) return k;
  }
  return -1;
}

// kProduct[i][j] = index of monomial i times monomial j, -1 above degree 3 (computed at compile
// time, so the products below are plain table lookups).
struct ProductTable {
  int at[20][20];
};
constexpr ProductTable make_products() {
  ProductTable t{};
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 20; ++j) {
      const int a = kExpo[i][0] + kExpo[j][0];
      const int b = kExpo[i][1] + kExpo[j][1];
      const int c = kExpo[i][2] + kExpo[j][2];
      t.at[i][j] = a + b + c > 3 ? -1 : mono(a, b, c);
    }
  }
  return t;
}
constexpr ProductTable kProduct = make_products();

Poly mul(const Poly& p, const Poly& q) {
  Poly r{};
  for (int i = 0; i < 20; ++i) {
    const double pi = p[static_cast<std::size_t>(i)];
    if (pi == 0.0) continue;
    for (int j = 0; j < 20; ++j) {
      const double qj = q[static_cast<std::size_t>(j)];
      if (qj == 0.0) continue;
      const int m = kProduct.at[i][j];
      if (m < 0) continue;  // never happens for our degrees
      r[static_cast<std::size_t>(m)] += pi * qj;
    }
  }
  return r;
}

Poly add(const Poly& p, const Poly& q, double s = 1.0) {
  Poly r{};
  for (std::size_t i = 0; i < 20; ++i) r[i] = p[i] + s * q[i];
  return r;
}

Poly scale(const Poly& p, double s) {
  Poly r{};
  for (std::size_t i = 0; i < 20; ++i) r[i] = s * p[i];
  return r;
}

double sampson_sq_e(const Eigen::Matrix3d& E, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  const Eigen::Vector3d ha(a.x(), a.y(), 1.0);
  const Eigen::Vector3d hb(b.x(), b.y(), 1.0);
  const Eigen::Vector3d Ea = E * ha;
  const Eigen::Vector3d Etb = E.transpose() * hb;
  const double e = hb.dot(Ea);
  const double d = Ea(0) * Ea(0) + Ea(1) * Ea(1) + Etb(0) * Etb(0) + Etb(1) * Etb(1);
  return d > 0.0 ? e * e / d : 0.0;
}

}  // namespace

std::vector<Eigen::Matrix3d> five_point(const Eigen::Vector2d x1n[5],
                                        const Eigen::Vector2d x2n[5]) {
  std::vector<Eigen::Matrix3d> out;
  // 1. Null space of the 5 x 9 system (row-major E: e00 e01 e02 e10 ... e22).
  Eigen::Matrix<double, 5, 9> Q;
  for (int k = 0; k < 5; ++k) {
    const double u1 = x1n[k].x();
    const double v1 = x1n[k].y();
    const double u2 = x2n[k].x();
    const double v2 = x2n[k].y();
    Q.row(k) << u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0;
  }
  // Null space from a pivoted QR of Q^T (9 x 5): the last 4 columns of its orthogonal factor are
  // orthogonal to every row of Q (several times cheaper than an SVD here).
  const Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 5>> qr(Q.transpose());
  const auto& R = qr.matrixR();
  if (!(std::fabs(R(4, 4)) >= 1e-12 * std::fabs(R(0, 0)))) return out;  // rank < 5: degenerate
  const Eigen::Matrix<double, 9, 9> H = qr.householderQ();
  const Eigen::Matrix<double, 9, 4> N = H.rightCols<4>();  // X Y Z W

  // E as a 3x3 matrix of linear polynomials.
  std::array<std::array<Poly, 3>, 3> E{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Poly& p = E[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
      p.fill(0.0);
      const int k = 3 * r + c;
      p[static_cast<std::size_t>(mono(1, 0, 0))] = N(k, 0);
      p[static_cast<std::size_t>(mono(0, 1, 0))] = N(k, 1);
      p[static_cast<std::size_t>(mono(0, 0, 1))] = N(k, 2);
      p[static_cast<std::size_t>(mono(0, 0, 0))] = N(k, 3);
    }
  }
  const auto at = [&](int r, int c) -> const Poly& {
    return E[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
  };

  // 2. Constraints. EEt = E E^T (degree 2).
  std::array<std::array<Poly, 3>, 3> EEt{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Poly acc{};
      for (int k = 0; k < 3; ++k) acc = add(acc, mul(at(r, k), at(c, k)));
      EEt[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = acc;
    }
  }
  const Poly tr = add(add(EEt[0][0], EEt[1][1]), EEt[2][2]);
  Eigen::Matrix<double, 10, 20> M;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Poly acc{};
      for (int k = 0; k < 3; ++k) {
        acc =
            add(acc, mul(EEt[static_cast<std::size_t>(r)][static_cast<std::size_t>(k)], at(k, c)));
      }
      const Poly row = add(scale(acc, 2.0), mul(tr, at(r, c)), -1.0);
      for (int m = 0; m < 20; ++m) M(3 * r + c, m) = row[static_cast<std::size_t>(m)];
    }
  }
  const Poly det =
      add(add(mul(at(0, 0), add(mul(at(1, 1), at(2, 2)), mul(at(1, 2), at(2, 1)), -1.0)),
              mul(at(0, 1), add(mul(at(1, 2), at(2, 0)), mul(at(1, 0), at(2, 2)), -1.0))),
          mul(at(0, 2), add(mul(at(1, 0), at(2, 1)), mul(at(1, 1), at(2, 0)), -1.0)));
  for (int m = 0; m < 20; ++m) M(9, m) = det[static_cast<std::size_t>(m)];

  // 3. Eliminate the 10 cubic monomials: [A | B] -> [I | A^-1 B].
  const Eigen::Matrix<double, 10, 10> A = M.leftCols<10>();
  const Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(A);
  if (!lu.isInvertible()) return out;
  const Eigen::Matrix<double, 10, 10> B = lu.solve(M.rightCols<10>());

  // Action matrix of x on b = [x^2 xy xz y^2 yz z^2 x y z 1]:
  // x b = [x^3 x^2y x^2z xy^2 xyz xz^2 | x^2 xy xz x]; the first six are -B rows 0..5 applied to b.
  Eigen::Matrix<double, 10, 10> At = Eigen::Matrix<double, 10, 10>::Zero();
  for (int r = 0; r < 6; ++r) At.row(r) = -B.row(r);
  At(6, 0) = 1.0;  // x * x  = x^2
  At(7, 1) = 1.0;  // x * y  = xy
  At(8, 2) = 1.0;  // x * z  = xz
  At(9, 6) = 1.0;  // x * 1  = x
  const Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> es(At);
  if (es.info() != Eigen::Success) return out;
  for (int k = 0; k < 10; ++k) {
    const std::complex<double> lambda = es.eigenvalues()(k);
    if (std::fabs(lambda.imag()) > 1e-8 * std::max(1.0, std::fabs(lambda.real()))) continue;
    const Eigen::Matrix<std::complex<double>, 10, 1> v = es.eigenvectors().col(k);
    if (std::abs(v(9)) < 1e-14) continue;
    const double x = (v(6) / v(9)).real();
    const double y = (v(7) / v(9)).real();
    const double z = (v(8) / v(9)).real();
    const Eigen::Matrix<double, 9, 1> e = x * N.col(0) + y * N.col(1) + z * N.col(2) + N.col(3);
    Eigen::Matrix3d Em;
    Em << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
    const double n = Em.norm();
    if (n > 0.0) out.push_back(Em / n);
  }
  return out;
}

EssentialRansacResult ransac_essential(const std::vector<Eigen::Vector2d>& x1n,
                                       const std::vector<Eigen::Vector2d>& x2n,
                                       const EssentialRansacParams& p) {
  EssentialRansacResult r;
  const std::size_t n = x1n.size();
  r.inlier.assign(n, 0);
  if (n < 5 || x2n.size() != n) return r;
  const double thr = p.threshold_px / p.focal;
  const double thr2 = thr * thr;
  eval::Rng rng(p.seed);
  std::vector<std::size_t> perm(n);
  double best_cost = std::numeric_limits<double>::infinity();
  Eigen::Matrix3d best_E = Eigen::Matrix3d::Zero();
  int limit = p.max_iters;
  int it = 0;
  for (; it < limit; ++it) {
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    Eigen::Vector2d a[5];
    Eigen::Vector2d b[5];
    for (std::size_t k = 0; k < 5; ++k) {
      std::swap(perm[k], perm[k + rng.index(n - k)]);
      a[k] = x1n[perm[k]];
      b[k] = x2n[perm[k]];
    }
    for (const Eigen::Matrix3d& E : five_point(a, b)) {
      double cost = 0.0;
      int inl = 0;
      for (std::size_t k = 0; k < n && cost < best_cost; ++k) {
        const double d = sampson_sq_e(E, x1n[k], x2n[k]);
        cost += std::min(d, thr2);
        inl += d < thr2 ? 1 : 0;
      }
      if (!(cost < best_cost)) continue;
      best_cost = cost;
      best_E = E;
      const double w5 = std::pow(static_cast<double>(inl) / static_cast<double>(n), 5);
      if (w5 >= 1.0) {
        limit = it + 1;
      } else if (w5 > 0.0) {
        const double need = std::log(1.0 - p.confidence) / std::log(1.0 - w5);
        if (need < limit) limit = std::max(it + 1, static_cast<int>(std::ceil(need)));
      }
    }
  }
  r.iterations = it;
  r.E = best_E;
  for (std::size_t k = 0; k < n; ++k) {
    const bool in = best_E.norm() > 0.0 && sampson_sq_e(best_E, x1n[k], x2n[k]) < thr2;
    r.inlier[k] = in ? 1 : 0;
    r.inliers += in ? 1 : 0;
  }
  return r;
}

}  // namespace sfm::matching
