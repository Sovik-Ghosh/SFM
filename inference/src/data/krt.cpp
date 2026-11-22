#include "sfm/data/krt.hpp"

#include <cmath>
#include <string>

#include <Eigen/LU>
#include <Eigen/QR>

#include "sfm/core/error.hpp"

namespace sfm::data {

Mat34 compose_projection(const Krt& krt) {
  Mat34 P;
  P.leftCols<3>() = krt.K * krt.R;
  P.col(3) = krt.K * krt.t;
  return P;
}

// RQ via QR (Hartley & Zisserman 6.2.4). With J the reversal permutation (J = J^T = J^-1, and
// J U J lower-triangular for upper-triangular U):
//   (J M)^T = Q U                      QR of the flipped, transposed block
//   J M     = U^T Q^T                  transpose
//   M       = J U^T Q^T = (J U^T J)(J Q^T)
// J U^T J is upper-triangular (U^T is lower, J..J flips it back) and J Q^T is orthonormal.
Krt decompose_projection(const Mat34& P_in) {
  Mat34 P = P_in;
  Eigen::Matrix3d M = P.leftCols<3>();

  // 1. Singular M = not a projective camera. The tolerance is relative: det scales with the
  //    cube of the entries, so compare against ||M||^3 (P scaled by 1e-3 must still pass).
  const double size = M.norm();
  const double det = M.determinant();
  if (!(size > 0.0) || std::abs(det) <= 1e-12 * size * size * size) {
    throw IoError("decompose_projection: left 3x3 block of P is singular (det " +
                  std::to_string(det) + ")");
  }

  // 2. P is defined up to any non-zero scale, sign included. det M = lambda^3 det K det R; after
  //    making det M > 0, diag(K) > 0 (step 4) forces det R = +1: a rotation, not a reflection.
  if (det < 0.0) {
    P = -P;
    M = -M;
  }

  // 3. RQ decomposition M = K R from a QR decomposition (see above).
  Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
  J(0, 2) = J(1, 1) = J(2, 0) = 1.0;
  const Eigen::HouseholderQR<Eigen::Matrix3d> qr((J * M).transpose());
  const Eigen::Matrix3d Q = qr.householderQ();
  const Eigen::Matrix3d U = qr.matrixQR().triangularView<Eigen::Upper>();  // exact zeros below
  Eigen::Matrix3d K = J * U.transpose() * J;
  Eigen::Matrix3d R = J * Q.transpose();

  // 4. Sign fix: K R = (K D)(D R) for D = diag(+-1), D^2 = I. Choose D so that diag(K) > 0.
  const Eigen::Vector3d signs =
      K.diagonal().unaryExpr([](double v) { return v < 0.0 ? -1.0 : 1.0; });
  K = K * signs.asDiagonal();
  R = signs.asDiagonal() * R;

  // 5. p4 = lambda K t with the same (unnormalised) K, so solve before rescaling it.
  Krt out;
  out.t = K.triangularView<Eigen::Upper>().solve(Eigen::Vector3d(P.col(3)));

  // 6. Normalise the scale: K(2,2) = 1. R is scale-free; t already carries the right scale.
  out.K = K / K(2, 2);
  out.R = R;
  return out;
}

}  // namespace sfm::data
