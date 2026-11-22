#pragma once
// Device side of the GPU bundle adjustment (no Eigen: nvcc warns on Eigen's constexpr host
// functions and CI builds with -Werror). Every sum is reduced in a fixed order (one thread per
// camera / point / camera pair walking a host-built list), so results are deterministic.

#include <memory>
#include <vector>

namespace sfm::ba::cuda {

// Problem structure, fixed during one solve (built once on the host).
struct Structure {
  int cameras = 0;
  int points = 0;
  std::vector<int> obs_cam;     // per observation
  std::vector<int> obs_pt;      // per observation
  std::vector<double> uv;       // 2 per observation
  std::vector<double> K;        // 9 per camera, row-major
  std::vector<int> obs_sensor;  // per observation: rig sensor, -1 = none (must be set)
  std::vector<double> sensor;   // 21 per sensor: K 9, R 9 (row-major), t 3
  std::vector<int> cam_ptr;     // CSR camera -> observations (cameras + 1)
  std::vector<int> cam_obs;
  std::vector<int> pt_ptr;  // CSR point -> observations (points + 1)
  std::vector<int> pt_obs;
  std::vector<int> free_col;  // 6 per camera: column in the reduced system, -1 = fixed
  int n_free = 0;
  // Camera-pair blocks of the Schur complement: pair k = (pair_a[k], pair_b[k]); its terms are
  // (term_oa, term_ob) for term_ptr[k] <= i < term_ptr[k + 1] (observation of camera a and of
  // camera b of one point), in increasing point order.
  std::vector<int> pair_a;
  std::vector<int> pair_b;
  std::vector<int> term_ptr;
  std::vector<int> term_oa;
  std::vector<int> term_ob;
};

// Builds cam/pt CSR and the pair terms from obs_cam / obs_pt (the other fields must be set).
void finish_structure(Structure& s);

class Device {
 public:
  // Takes the structure (moved in: the pair terms are the largest host array) and frees the
  // arrays that are only uploaded.
  explicit Device(Structure s);
  int observations() const;
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // R: 9 per camera (row-major), t: 3 per camera, X: 3 per point.
  void set_params(const std::vector<double>& R, const std::vector<double>& t,
                  const std::vector<double>& X);
  double cost(double huber);
  // Jacobians and the blocks U, V, W, gc, gp at the current parameters.
  void linearize(double huber);
  // Blocks of the last linearisation (for tests and the predicted decrease): U 36 / camera,
  // V 9 / point, W 18 / observation (row-major 6x3), gc 6 / camera, gp 3 / point.
  void download(std::vector<double>* U, std::vector<double>* V, std::vector<double>* W,
                std::vector<double>* gc, std::vector<double>* gp) const;
  // Damped Schur solve: dc 6 / camera (0 for fixed parameters), dp 3 / point. False if the
  // reduced system is not positive definite.
  bool solve(double lambda, std::vector<double>& dc, std::vector<double>& dp);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sfm::ba::cuda
