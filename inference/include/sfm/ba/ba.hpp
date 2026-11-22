#pragma once
// Sparse bundle adjustment: Levenberg-Marquardt over camera poses and 3D points with the Schur
// complement (reduced camera system), Huber loss, K fixed.
//
// Parameters: per camera 6 (w, dt) with the update R <- exp([w]_x) R, t <- t + dt; per point 3.
// Residual r = project(K, R X + t) - uv (2 per observation), Huber weight
// w = min(1, delta / ||r||) (IRLS), cost = 1/2 sum rho(||r||), rho(s) = s^2 (s <= delta),
// 2 delta s - delta^2 (s > delta).
//
// Rigs (P9b): a "camera" can be one moment (frame) of a rig of sensors fixed to each other. An
// observation through sensor s then projects with K_s and the pose T_s * T_c, where T_s = (R_s,
// t_s) is the sensor's fixed offset (x_s = R_s x_c + t_s). With the update above applied to T_c the
// composed pose moves by w' = R_s w, dt' = R_s dt, so the camera Jacobian is the one of the
// composed pose times blockdiag(R_s, R_s); the Schur structure is unchanged, with one block per
// frame.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sfm::ba {

struct Observation {
  int camera = 0;
  int point = 0;
  Eigen::Vector2d uv = Eigen::Vector2d::Zero();
  int sensor = -1;  // -1: the camera's own K and pose; else Problem::sensors[sensor]
};

// A rig sensor: intrinsics and the fixed offset from the frame pose (x_sensor = R x_frame + t).
struct Sensor {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

struct Problem {
  std::vector<Eigen::Matrix3d> K;
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Vector3d> t;
  std::vector<Eigen::Vector3d> X;
  std::vector<Observation> obs;
  // Per camera, which of the 6 parameters (w0, w1, w2, t0, t1, t2) are held constant. The gauge:
  // one camera fully fixed (frame) and one translation component of another (scale).
  std::vector<std::array<bool, 6>> fixed;
  std::vector<Sensor> sensors;  // empty unless observations use a rig sensor

  std::size_t cameras() const { return R.size(); }
  std::size_t points() const { return X.size(); }
};

struct Options {
  double huber_px = 2.0;
  int max_iters = 50;
  double initial_lambda = 1e-3;
  // Stop when an accepted step lowers the cost by less than this share (Ceres default). 1e-9 made
  // almost every BA run to max_iters; playground rig: BA 283 -> 74 s, accuracy unchanged (P9c).
  double function_tolerance = 1e-6;
};

struct Report {
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int iterations = 0;  // LM iterations (accepted + rejected steps)
  int accepted = 0;
  std::vector<double> cost_log;  // cost after every accepted step (first entry: initial)
  bool converged = false;
};

// Robust cost of the current parameters.
double cost(const Problem& p, double huber_px);

// 2x6 camera and 2x3 point Jacobians of the residual of one observation (analytic).
void jacobians(const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
               const Eigen::Vector3d& X, Eigen::Matrix<double, 2, 6>& Jc,
               Eigen::Matrix<double, 2, 3>& Jx);

// K, R, t an observation projects with: the camera's, or the sensor offset composed with it.
struct ObsPose {
  Eigen::Matrix3d K;
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
};
ObsPose observation_pose(const Problem& p, const Observation& o);

// Residual (pixels) and Jacobians of one observation, rig sensors included (Jc w.r.t. the update
// of Problem::R / t of o.camera).
Eigen::Vector2d residual(const Problem& p, const Observation& o);
void observation_jacobians(const Problem& p, const Observation& o, Eigen::Matrix<double, 2, 6>& Jc,
                           Eigen::Matrix<double, 2, 3>& Jx);

// Normal equations of one LM step in block form: H = [U W; W^T V], g = [gc; gp]. U is the dense
// camera block (6 nc x 6 nc, block diagonal), V the 3x3 point blocks, W the camera-point blocks.
struct NormalEquations {
  std::vector<Eigen::Matrix<double, 6, 6>> U;  // per camera
  std::vector<Eigen::Matrix3d> V;              // per point
  std::vector<Eigen::Matrix<double, 6, 3>> W;  // per observation
  std::vector<Eigen::Matrix<double, 6, 1>> gc;
  std::vector<Eigen::Vector3d> gp;
};
NormalEquations build_normal_equations(const Problem& p, double huber_px);

// Solves (H + damping) [dc; dp] = -g by the Schur complement on the camera block:
//   S = U - W V^-1 W^T,  b = -gc + W V^-1 gp,  S dc = b (dense Cholesky),  dp = V^-1 (-gp - W^T dc)
// with damping lambda * diag (Marquardt) added to U and V. Fixed camera parameters get
// dc = 0. Returns false if S is not positive definite.
bool schur_solve(const Problem& p, const NormalEquations& ne, double lambda,
                 std::vector<Eigen::Matrix<double, 6, 1>>& dc, std::vector<Eigen::Vector3d>& dp);

// What the Levenberg-Marquardt loop needs from a linear-algebra backend (CPU: the functions
// above; GPU: sfm/ba/ba_cuda.hpp). The loop itself, and so the step acceptance and damping, is
// shared.
class LmBackend {
 public:
  virtual ~LmBackend() = default;
  virtual double cost(const Problem& p, double huber_px) = 0;
  // Linearise at p (normal equations kept by the backend).
  virtual void linearize(const Problem& p, double huber_px) = 0;
  // Damped Schur solve of the last linearisation; false if the reduced system is not positive
  // definite.
  virtual bool solve(const Problem& p, double lambda, std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                     std::vector<Eigen::Vector3d>& dp) = 0;
  // 1/2 (-g.d + lambda d.D.d) of the last linearisation (D = diag of U and V).
  virtual double predicted_decrease(double lambda,
                                    const std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                                    const std::vector<Eigen::Vector3d>& dp) = 0;
};

// Levenberg-Marquardt with Nielsen's damping update (gain ratio of actual to predicted cost
// decrease). Stops on max_iters, a relative decrease below function_tolerance, or when no damping
// gives a decrease. Updates p in place.
Report solve_lm(Problem& p, const Options& o, LmBackend& backend);

// solve_lm with the CPU backend.
Report solve(Problem& p, const Options& o);

}  // namespace sfm::ba
