// d residual / d parameters. x_c = R X + t = (a, b, c); pixel u = k00 a/c + k01 b/c + k02,
// v = k11 b/c + k12. With the left perturbation R <- exp([w]_x) R:
//   d x_c / d w = -[R X]_x,  d x_c / d t = I,  d x_c / d X = R.
#include "sfm/ba/ba.hpp"

namespace sfm::ba {

void jacobians(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
               const Eigen::Vector3d& X, Eigen::Matrix<double, 2, 6>& Jc,
               Eigen::Matrix<double, 2, 3>& Jx) {
  const Eigen::Vector3d RX = R * X;
  const Eigen::Vector3d xc = RX + t;
  const double iz = 1.0 / xc.z();
  const double iz2 = iz * iz;
  Eigen::Matrix<double, 2, 3> Jp;  // d pixel / d x_c
  Jp << K(0, 0) * iz, K(0, 1) * iz, -(K(0, 0) * xc.x() + K(0, 1) * xc.y()) * iz2,  //
      0.0, K(1, 1) * iz, -K(1, 1) * xc.y() * iz2;
  Eigen::Matrix3d skew_rx;
  skew_rx << 0, -RX.z(), RX.y(), RX.z(), 0, -RX.x(), -RX.y(), RX.x(), 0;
  Jc.leftCols<3>() = -Jp * skew_rx;
  Jc.rightCols<3>() = Jp;
  Jx = Jp * R;
}

}  // namespace sfm::ba
