#pragma once
// Bundle adjustment with the linear algebra on the GPU (P6 in the original plan, released as
// v0.5.0): residuals and Jacobians per observation, the blocks U, V, W, the Schur complement
// S = U - W V^-1 W^T, a dense Cholesky solve of S and the point back-substitution all run on the
// device; the Levenberg-Marquardt loop (step acceptance, damping) is the CPU's (solve_lm).
// Every sum has a fixed order, so results are deterministic; they equal the CPU's to rounding
// (different summation order), not bit for bit.

#include <memory>
#include <vector>

#include "sfm/ba/ba.hpp"

namespace sfm::ba {

class CudaBackend final : public LmBackend {
 public:
  explicit CudaBackend(const Problem& structure);  // uploads the observations; params later
  ~CudaBackend() override;
  CudaBackend(const CudaBackend&) = delete;
  CudaBackend& operator=(const CudaBackend&) = delete;

  double cost(const Problem& p, double huber_px) override;
  void linearize(const Problem& p, double huber_px) override;
  bool solve(const Problem& p, double lambda, std::vector<Eigen::Matrix<double, 6, 1>>& dc,
             std::vector<Eigen::Vector3d>& dp) override;
  double predicted_decrease(double lambda, const std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                            const std::vector<Eigen::Vector3d>& dp) override;

  // The last linearisation as NormalEquations (for comparison with build_normal_equations).
  NormalEquations normal_equations() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// solve_lm with the GPU backend.
Report solve_cuda(Problem& p, const Options& o);

}  // namespace sfm::ba
