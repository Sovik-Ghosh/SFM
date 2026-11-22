// CudaBackend: Problem (Eigen) <-> the device arrays of ba_device.cuh.
#include "sfm/ba/ba_cuda.hpp"

#include <utility>

#include "ba_device.cuh"

namespace sfm::ba {

struct CudaBackend::Impl {
  cuda::Structure s;
  std::unique_ptr<cuda::Device> dev;
  std::vector<double> U, V, gc, gp;  // last linearisation (host copies)

  void upload(const Problem& p) const {
    std::vector<double> R(9 * p.cameras());
    std::vector<double> t(3 * p.cameras());
    std::vector<double> X(3 * p.points());
    for (std::size_t c = 0; c < p.cameras(); ++c) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R[9 * c + static_cast<std::size_t>(3 * i + j)] = p.R[c](i, j);
        t[3 * c + static_cast<std::size_t>(i)] = p.t[c](i);
      }
    }
    for (std::size_t x = 0; x < p.points(); ++x) {
      for (int i = 0; i < 3; ++i) X[3 * x + static_cast<std::size_t>(i)] = p.X[x](i);
    }
    dev->set_params(R, t, X);
  }
};

CudaBackend::CudaBackend(const Problem& p) : impl_(std::make_unique<Impl>()) {
  cuda::Structure& s = impl_->s;
  s.cameras = static_cast<int>(p.cameras());
  s.points = static_cast<int>(p.points());
  for (const Observation& o : p.obs) {
    s.obs_cam.push_back(o.camera);
    s.obs_pt.push_back(o.point);
    s.obs_sensor.push_back(o.sensor);
    s.uv.push_back(o.uv.x());
    s.uv.push_back(o.uv.y());
  }
  for (std::size_t c = 0; c < p.cameras(); ++c) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) s.K.push_back(p.K[c](i, j));
    }
    for (std::size_t k = 0; k < 6; ++k) s.free_col.push_back(p.fixed[c][k] ? -1 : s.n_free++);
  }
  for (const Sensor& r : p.sensors) {
    for (const Eigen::Matrix3d* M : {&r.K, &r.R}) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) s.sensor.push_back((*M)(i, j));
      }
    }
    for (int i = 0; i < 3; ++i) s.sensor.push_back(r.t(i));
  }
  cuda::finish_structure(s);
  impl_->dev = std::make_unique<cuda::Device>(std::move(s));  // s keeps the counts only
}

CudaBackend::~CudaBackend() = default;

double CudaBackend::cost(const Problem& p, double huber_px) {
  impl_->upload(p);
  return impl_->dev->cost(huber_px);
}

void CudaBackend::linearize(const Problem& p, double huber_px) {
  impl_->upload(p);
  impl_->dev->linearize(huber_px);
  impl_->dev->download(&impl_->U, &impl_->V, nullptr, &impl_->gc, &impl_->gp);
}

bool CudaBackend::solve(const Problem& p, double lambda,
                        std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                        std::vector<Eigen::Vector3d>& dp) {
  std::vector<double> c;
  std::vector<double> x;
  if (!impl_->dev->solve(lambda, c, x)) return false;
  dc.assign(p.cameras(), Eigen::Matrix<double, 6, 1>::Zero());
  dp.assign(p.points(), Eigen::Vector3d::Zero());
  for (std::size_t k = 0; k < p.cameras(); ++k) {
    for (int a = 0; a < 6; ++a) dc[k](a) = c[6 * k + static_cast<std::size_t>(a)];
  }
  for (std::size_t k = 0; k < p.points(); ++k) {
    for (int a = 0; a < 3; ++a) dp[k](a) = x[3 * k + static_cast<std::size_t>(a)];
  }
  return true;
}

double CudaBackend::predicted_decrease(double lambda,
                                       const std::vector<Eigen::Matrix<double, 6, 1>>& dc,
                                       const std::vector<Eigen::Vector3d>& dp) {
  const Impl& m = *impl_;
  double gd = 0.0;
  double dDd = 0.0;
  for (std::size_t c = 0; c < dc.size(); ++c) {
    for (std::size_t a = 0; a < 6; ++a) {
      const double d = dc[c](static_cast<Eigen::Index>(a));
      gd += m.gc[6 * c + a] * d;
      dDd += m.U[36 * c + 7 * a] * d * d;
    }
  }
  for (std::size_t x = 0; x < dp.size(); ++x) {
    for (std::size_t a = 0; a < 3; ++a) {
      const double d = dp[x](static_cast<Eigen::Index>(a));
      gd += m.gp[3 * x + a] * d;
      dDd += m.V[9 * x + 4 * a] * d * d;
    }
  }
  return 0.5 * (-gd + lambda * dDd);
}

NormalEquations CudaBackend::normal_equations() const {
  std::vector<double> U;
  std::vector<double> V;
  std::vector<double> W;
  std::vector<double> gc;
  std::vector<double> gp;
  impl_->dev->download(&U, &V, &W, &gc, &gp);
  NormalEquations ne;
  const auto nc = static_cast<std::size_t>(impl_->s.cameras);
  const auto np = static_cast<std::size_t>(impl_->s.points);
  const auto no = static_cast<std::size_t>(impl_->dev->observations());
  ne.U.resize(nc);
  ne.gc.resize(nc);
  for (std::size_t c = 0; c < nc; ++c) {
    for (int i = 0; i < 6; ++i) {
      ne.gc[c](i) = gc[6 * c + static_cast<std::size_t>(i)];
      for (int j = 0; j < 6; ++j) ne.U[c](i, j) = U[36 * c + static_cast<std::size_t>(6 * i + j)];
    }
  }
  ne.V.resize(np);
  ne.gp.resize(np);
  for (std::size_t x = 0; x < np; ++x) {
    for (int i = 0; i < 3; ++i) {
      ne.gp[x](i) = gp[3 * x + static_cast<std::size_t>(i)];
      for (int j = 0; j < 3; ++j) ne.V[x](i, j) = V[9 * x + static_cast<std::size_t>(3 * i + j)];
    }
  }
  ne.W.resize(no);
  for (std::size_t k = 0; k < no; ++k) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 3; ++j) ne.W[k](i, j) = W[18 * k + static_cast<std::size_t>(3 * i + j)];
    }
  }
  return ne;
}

Report solve_cuda(Problem& p, const Options& o) {
  CudaBackend backend(p);
  return solve_lm(p, o, backend);
}

}  // namespace sfm::ba
