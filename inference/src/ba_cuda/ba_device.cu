// GPU bundle adjustment kernels. See ba_device.cuh and sfm/ba/ba.hpp for the maths.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <cusolverDn.h>
#include <unistd.h>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"

#include "ba_device.cuh"

namespace sfm::ba::cuda {

namespace {

// Physical memory of the host in bytes (8 GiB if the system does not say).
std::size_t host_memory_bytes() {
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page <= 0) return std::size_t{8} << 30;
  return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page);
}

}  // namespace

namespace {

constexpr int kBlock = 256;

int blocks(int n) { return (n + kBlock - 1) / kBlock; }

// Residual, Huber weight and Jacobians of one observation (left perturbation of R). Through a rig
// sensor (obs_sensor >= 0; sensor: K 9, R 9, t 3 per sensor) the pose is the sensor offset composed
// with the camera's, and the camera Jacobian is multiplied by blockdiag(R_s, R_s) (sfm/ba/ba.hpp).
__global__ void obs_kernel(int nobs, const int* obs_cam, const int* obs_pt, const int* obs_sensor,
                           const double* uv, const double* K, const double* sensor, const double* R,
                           const double* t, const double* X, double huber, double* Jc, double* Jx,
                           double* r, double* w, double* rho) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nobs) return;
  const int c = obs_cam[k];
  const int p = obs_pt[k];
  const int sn = obs_sensor[k];
  const double* Rs = sn >= 0 ? sensor + 21 * sn + 9 : nullptr;
  double Rc[9];
  double tc[3];
  const double* Kc = K + 9 * c;
  if (sn < 0) {
    for (int i = 0; i < 9; ++i) Rc[i] = R[9 * c + i];
    for (int i = 0; i < 3; ++i) tc[i] = t[3 * c + i];
  } else {
    Kc = sensor + 21 * sn;
    const double* ts = sensor + 21 * sn + 18;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        Rc[3 * i + j] = Rs[3 * i] * R[9 * c + j] + Rs[3 * i + 1] * R[9 * c + 3 + j] +
                        Rs[3 * i + 2] * R[9 * c + 6 + j];
      }
      tc[i] = Rs[3 * i] * t[3 * c] + Rs[3 * i + 1] * t[3 * c + 1] + Rs[3 * i + 2] * t[3 * c + 2] +
              ts[i];
    }
  }
  const double* Xp = X + 3 * p;
  double RX[3];
  for (int i = 0; i < 3; ++i)
    RX[i] = Rc[3 * i] * Xp[0] + Rc[3 * i + 1] * Xp[1] + Rc[3 * i + 2] * Xp[2];
  const double a = RX[0] + tc[0];
  const double b = RX[1] + tc[1];
  const double cz = RX[2] + tc[2];
  const double iz = 1.0 / cz;
  const double u = (Kc[0] * a + Kc[1] * b + Kc[2] * cz) * iz;
  const double v = (Kc[4] * b + Kc[5] * cz) * iz;
  const double r0 = u - uv[2 * k];
  const double r1 = v - uv[2 * k + 1];
  r[2 * k] = r0;
  r[2 * k + 1] = r1;
  const double s = sqrt(r0 * r0 + r1 * r1);
  w[k] = s <= huber ? 1.0 : huber / s;
  rho[k] = s <= huber ? s * s : 2.0 * huber * s - huber * huber;
  // d pixel / d x_c
  const double iz2 = iz * iz;
  const double P[2][3] = {{Kc[0] * iz, Kc[1] * iz, -(Kc[0] * a + Kc[1] * b) * iz2},
                          {0.0, Kc[4] * iz, -Kc[4] * b * iz2}};
  // -[RX]_x
  const double S[3][3] = {{0, RX[2], -RX[1]}, {-RX[2], 0, RX[0]}, {RX[1], -RX[0], 0}};
  for (int i = 0; i < 2; ++i) {
    double Jw[3];
    for (int j = 0; j < 3; ++j) {
      Jw[j] = P[i][0] * S[0][j] + P[i][1] * S[1][j] + P[i][2] * S[2][j];
      Jx[6 * k + 3 * i + j] = P[i][0] * Rc[j] + P[i][1] * Rc[3 + j] + P[i][2] * Rc[6 + j];
    }
    for (int j = 0; j < 3; ++j) {
      if (sn < 0) {
        Jc[12 * k + 6 * i + j] = Jw[j];
        Jc[12 * k + 6 * i + 3 + j] = P[i][j];
      } else {
        Jc[12 * k + 6 * i + j] = Jw[0] * Rs[j] + Jw[1] * Rs[3 + j] + Jw[2] * Rs[6 + j];
        Jc[12 * k + 6 * i + 3 + j] = P[i][0] * Rs[j] + P[i][1] * Rs[3 + j] + P[i][2] * Rs[6 + j];
      }
    }
  }
}

constexpr int kChunks = 8;  // a camera's observation list is summed in 8 chunks, then in order

// Chunk range of list [lo, hi) for chunk `part`.
__device__ void chunk(int lo, int hi, int part, int& b, int& e) {
  const int len = hi - lo;
  b = lo + static_cast<int>((static_cast<long long>(len) * part) / kChunks);
  e = lo + static_cast<int>((static_cast<long long>(len) * (part + 1)) / kChunks);
}

// Grid (camera, chunk), one thread per output element (36 of U, 6 of gc): partial sums over the
// chunk of the camera's observations, in list order. Summed by cam_sum_kernel in chunk order.
__global__ void cam_kernel(const int* ptr, const int* list, const double* Jc, const double* r,
                           const double* w, double* partial) {
  const int c = blockIdx.x;
  const int part = blockIdx.y;
  const int e = threadIdx.x;
  if (e >= 42) return;
  int lo = 0;
  int hi = 0;
  chunk(ptr[c], ptr[c + 1], part, lo, hi);
  double acc = 0.0;
  if (e < 36) {
    const int a = e / 6;
    const int b = e % 6;
    for (int i = lo; i < hi; ++i) {
      const int k = list[i];
      const double* J = Jc + 12 * k;
      acc += w[k] * (J[a] * J[b] + J[6 + a] * J[6 + b]);
    }
  } else {
    const int a = e - 36;
    for (int i = lo; i < hi; ++i) {
      const int k = list[i];
      const double* J = Jc + 12 * k;
      acc += w[k] * (J[a] * r[2 * k] + J[6 + a] * r[2 * k + 1]);
    }
  }
  partial[(static_cast<long long>(c) * kChunks + part) * 42 + e] = acc;
}

__global__ void cam_sum_kernel(int nc, const double* partial, double* U, double* gc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc * 42) return;
  const int c = i / 42;
  const int e = i % 42;
  double acc = 0.0;
  for (int part = 0; part < kChunks; ++part)
    acc += partial[(static_cast<long long>(c) * kChunks + part) * 42 + e];
  if (e < 36) {
    U[36 * c + e] = acc;
  } else {
    gc[6 * c + e - 36] = acc;
  }
}

// Per point, in observation order: V = sum w Jx^T Jx, gp = sum w Jx^T r.
__global__ void pt_kernel(int np, const int* ptr, const int* list, const double* Jx,
                          const double* r, const double* w, double* V, double* gp) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= np) return;
  double v[9] = {};
  double g[3] = {};
  for (int i = ptr[p]; i < ptr[p + 1]; ++i) {
    const int k = list[i];
    const double* J = Jx + 6 * k;
    for (int a = 0; a < 3; ++a) {
      g[a] += w[k] * (J[a] * r[2 * k] + J[3 + a] * r[2 * k + 1]);
      for (int b = 0; b < 3; ++b) v[3 * a + b] += w[k] * (J[a] * J[b] + J[3 + a] * J[3 + b]);
    }
  }
  for (int a = 0; a < 9; ++a) V[9 * p + a] = v[a];
  for (int a = 0; a < 3; ++a) gp[3 * p + a] = g[a];
}

// Per observation: W = w Jc^T Jx (6x3).
__global__ void w_kernel(int nobs, const double* Jc, const double* Jx, const double* w, double* W) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nobs) return;
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 3; ++b) {
      W[18 * k + 3 * a + b] =
          w[k] * (Jc[12 * k + a] * Jx[6 * k + b] + Jc[12 * k + 6 + a] * Jx[6 * k + 3 + b]);
    }
  }
}

// Per point: Vinv = (V + lambda diag V + 1e-12 I)^-1 (adjugate; 0 if singular) and y = Vinv gp.
__global__ void vinv_kernel(int np, const double* V, const double* gp, double lambda, double* Vinv,
                            double* y, int* ok) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= np) return;
  double m[9];
  for (int a = 0; a < 9; ++a) m[a] = V[9 * p + a];
  for (int a = 0; a < 3; ++a) m[4 * a] += lambda * V[9 * p + 4 * a] + 1e-12;
  const double c00 = m[4] * m[8] - m[5] * m[7];
  const double c01 = m[5] * m[6] - m[3] * m[8];
  const double c02 = m[3] * m[7] - m[4] * m[6];
  const double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
  double inv[9] = {};
  const double scale = fabs(m[0]) + fabs(m[4]) + fabs(m[8]);
  const bool good = fabs(det) > 1e-30 * scale * scale * scale && isfinite(det);
  if (good) {
    const double id = 1.0 / det;
    inv[0] = c00 * id;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = c01 * id;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    inv[6] = c02 * id;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  }
  for (int a = 0; a < 9; ++a) Vinv[9 * p + a] = inv[a];
  for (int a = 0; a < 3; ++a) {
    y[3 * p + a] =
        inv[3 * a] * gp[3 * p] + inv[3 * a + 1] * gp[3 * p + 1] + inv[3 * a + 2] * gp[3 * p + 2];
  }
  ok[p] = good ? 1 : 0;
}

// Per observation: WV = W Vinv_point (6x3), reused by every Schur block the observation is in.
__global__ void wv_kernel(int nobs, const int* obs_pt, const double* W, const double* Vinv,
                          double* WV) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nobs) return;
  const double* Vi = Vinv + 9 * obs_pt[k];
  for (int a = 0; a < 6; ++a) {
    const double* Wa = W + 18 * k + 3 * a;
    for (int b = 0; b < 3; ++b)
      WV[18 * k + 3 * a + b] = Wa[0] * Vi[b] + Wa[1] * Vi[3 + b] + Wa[2] * Vi[6 + b];
  }
}

// Grid (camera, chunk), one thread per row: partial sums of W y_point over the chunk; b_sum_kernel
// adds them in chunk order to -gc (free rows only).
__global__ void b_kernel(const int* ptr, const int* list, const int* obs_pt, const double* W,
                         const double* y, double* partial) {
  const int c = blockIdx.x;
  const int part = blockIdx.y;
  const int a = threadIdx.x;
  if (a >= 6) return;
  int lo = 0;
  int hi = 0;
  chunk(ptr[c], ptr[c + 1], part, lo, hi);
  double acc = 0.0;
  for (int i = lo; i < hi; ++i) {
    const int k = list[i];
    const double* yp = y + 3 * obs_pt[k];
    acc +=
        W[18 * k + 3 * a] * yp[0] + W[18 * k + 3 * a + 1] * yp[1] + W[18 * k + 3 * a + 2] * yp[2];
  }
  partial[(static_cast<long long>(c) * kChunks + part) * 6 + a] = acc;
}

__global__ void b_sum_kernel(int nc, const double* partial, const double* gc, const int* free_col,
                             double* b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nc * 6) return;
  const int c = i / 6;
  const int a = i % 6;
  double acc = -gc[i];
  for (int part = 0; part < kChunks; ++part)
    acc += partial[(static_cast<long long>(c) * kChunks + part) * 6 + a];
  if (free_col[i] >= 0) b[free_col[i]] = acc;
}

// One block of 36 threads per camera pair: S(a, b) block = [a == b] (U + damping) - sum W_a Vinv
// W_b^T.
__global__ void s_kernel(const int* pair_a, const int* pair_b, const int* term_ptr,
                         const int* term_oa, const int* term_ob, const double* U, const double* W,
                         const double* WV, double lambda, const int* free_col, int n_free,
                         double* S) {
  const int k = blockIdx.x;
  const int i = threadIdx.x / 6;
  const int j = threadIdx.x % 6;
  if (threadIdx.x >= 36) return;
  const int ca = pair_a[k];
  const int cb = pair_b[k];
  double acc = 0.0;
  if (ca == cb) {
    acc = U[36 * ca + 6 * i + j];
    if (i == j) acc += lambda * U[36 * ca + 7 * i] + 1e-12;
  }
  for (int t = term_ptr[k]; t < term_ptr[k + 1]; ++t) {
    const double* A = WV + 18 * term_oa[t] + 3 * i;  // row i of W_a Vinv
    const double* B = W + 18 * term_ob[t] + 3 * j;   // row j of W_b
    acc -= A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
  }
  const int r = free_col[6 * ca + i];
  const int c = free_col[6 * cb + j];
  if (r >= 0 && c >= 0) S[static_cast<long long>(r) * n_free + c] = acc;
}

// In-place dense Cholesky S = L L^T (lower) and solve L L^T x = b, one block. ok = 0 if a pivot
// is not positive. n is a few hundred (6 x cameras): n sequential steps, each parallel.
__global__ void cholesky_solve_kernel(double* S, int n, double* b, int* ok) {
  extern __shared__ double col[];  // column k of L (n doubles)
  __shared__ int bad;
  if (threadIdx.x == 0) bad = 0;
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (threadIdx.x == 0) {
      const double d = S[static_cast<long long>(k) * n + k];
      if (!(d > 0.0)) {
        bad = 1;
      } else {
        S[static_cast<long long>(k) * n + k] = sqrt(d);
      }
    }
    __syncthreads();
    if (bad) break;
    const double dk = S[static_cast<long long>(k) * n + k];
    for (int i = k + 1 + threadIdx.x; i < n; i += blockDim.x) {
      const double l = S[static_cast<long long>(i) * n + k] / dk;
      S[static_cast<long long>(i) * n + k] = l;
      col[i] = l;
    }
    __syncthreads();
    // Trailing update of the lower triangle: S(i, j) -= L(i, k) L(j, k), k < j <= i. Threads as
    // a 32 x (blockDim / 32) grid: lanes along j (coalesced), warps along i.
    const int lane = static_cast<int>(threadIdx.x % 32);
    const int warp = static_cast<int>(threadIdx.x / 32);
    const int warps = static_cast<int>(blockDim.x / 32);
    for (int i = k + 1 + warp; i < n; i += warps) {
      const double li = col[i];
      double* row = S + static_cast<long long>(i) * n;
      for (int j = k + 1 + lane; j <= i; j += 32) row[j] -= li * col[j];
    }
    __syncthreads();
  }
  if (bad) {
    if (threadIdx.x == 0) *ok = 0;
    return;
  }
  // Forward L z = b, column by column: z_k = b_k / L_kk, then b_j -= L_jk z_k for j > k (parallel).
  for (int k = 0; k < n; ++k) {
    if (threadIdx.x == 0) b[k] /= S[static_cast<long long>(k) * n + k];
    __syncthreads();
    const double zk = b[k];
    for (int j = k + 1 + threadIdx.x; j < n; j += blockDim.x)
      b[j] -= S[static_cast<long long>(j) * n + k] * zk;
    __syncthreads();
  }
  // Backward L^T x = z: x_k = z_k / L_kk, then z_j -= L_kj x_k for j < k (row k of L, parallel).
  for (int k = n - 1; k >= 0; --k) {
    if (threadIdx.x == 0) b[k] /= S[static_cast<long long>(k) * n + k];
    __syncthreads();
    const double xk = b[k];
    for (int j = threadIdx.x; j < k; j += blockDim.x)
      b[j] -= S[static_cast<long long>(k) * n + j] * xk;
    __syncthreads();
  }
  if (threadIdx.x == 0) *ok = 1;
}

// Per point: dp = Vinv (-gp - sum_obs W^T dc_camera).
__global__ void dp_kernel(int np, const int* ptr, const int* list, const int* obs_cam,
                          const double* W, const double* Vinv, const double* gp, const double* x,
                          const int* free_col, double* dp) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= np) return;
  double rhs[3] = {-gp[3 * p], -gp[3 * p + 1], -gp[3 * p + 2]};
  for (int i = ptr[p]; i < ptr[p + 1]; ++i) {
    const int k = list[i];
    const int c = obs_cam[k];
    for (int a = 0; a < 6; ++a) {
      const int col = free_col[6 * c + a];
      if (col < 0) continue;
      const double d = x[col];
      for (int m = 0; m < 3; ++m) rhs[m] -= W[18 * k + 3 * a + m] * d;
    }
  }
  const double* Vi = Vinv + 9 * p;
  for (int m = 0; m < 3; ++m)
    dp[3 * p + m] = Vi[3 * m] * rhs[0] + Vi[3 * m + 1] * rhs[1] + Vi[3 * m + 2] * rhs[2];
}

// Fixed-order sum of n values: block partial sums (tree in shared memory) -> host adds in order.
__global__ void sum_kernel(const double* v, int n, double* partial) {
  __shared__ double s[kBlock];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? v[i] : 0.0;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

template <class T>
core::DeviceBuffer<T> make(const std::vector<T>& h, const core::Stream& s) {
  core::DeviceBuffer<T> d(std::max<std::size_t>(h.size(), 1));
  if (!h.empty()) d.upload(h.data(), h.size(), s);
  return d;
}

}  // namespace

void finish_structure(Structure& s) {
  const auto nobs = static_cast<int>(s.obs_cam.size());
  s.cam_ptr.assign(static_cast<std::size_t>(s.cameras) + 1, 0);
  s.pt_ptr.assign(static_cast<std::size_t>(s.points) + 1, 0);
  for (int k = 0; k < nobs; ++k) {
    ++s.cam_ptr[static_cast<std::size_t>(s.obs_cam[static_cast<std::size_t>(k)]) + 1];
    ++s.pt_ptr[static_cast<std::size_t>(s.obs_pt[static_cast<std::size_t>(k)]) + 1];
  }
  for (std::size_t c = 1; c < s.cam_ptr.size(); ++c) s.cam_ptr[c] += s.cam_ptr[c - 1];
  for (std::size_t p = 1; p < s.pt_ptr.size(); ++p) s.pt_ptr[p] += s.pt_ptr[p - 1];
  s.cam_obs.assign(static_cast<std::size_t>(nobs), 0);
  s.pt_obs.assign(static_cast<std::size_t>(nobs), 0);
  std::vector<int> cf(s.cam_ptr.begin(), s.cam_ptr.end() - 1);
  std::vector<int> pf(s.pt_ptr.begin(), s.pt_ptr.end() - 1);
  for (int k = 0; k < nobs; ++k) {  // observation order within each list
    s.cam_obs[static_cast<std::size_t>(
        cf[static_cast<std::size_t>(s.obs_cam[static_cast<std::size_t>(k)])]++)] = k;
    s.pt_obs[static_cast<std::size_t>(
        pf[static_cast<std::size_t>(s.obs_pt[static_cast<std::size_t>(k)])]++)] = k;
  }
  // Camera pairs sharing points, with their terms in increasing point order. A point seen by k
  // cameras gives k^2 terms (12 bytes each). Count first and refuse before allocating if the list
  // would not fit (half the host's memory, and int offsets); then fill the flat arrays in place,
  // so the peak is the list itself. Real tracks are 2-60 views; a point seen by hundreds of
  // cameras (a dense synthetic benchmark) is what blows up.
  // Only pairs of cameras with a free parameter are kept (P9c): S has rows and columns for free
  // parameters only, so a block with a fixed camera was summed and then dropped. In a local BA
  // most cameras are fixed (delivery_area rig: s_kernel was 64 % of all GPU time).
  const auto nc = static_cast<std::size_t>(s.cameras);
  std::vector<char> cam_free(nc, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    for (std::size_t a = 0; a < 6; ++a) {
      if (s.free_col[6 * c + a] >= 0) cam_free[c] = 1;
    }
  }
  std::size_t n_terms = 0;
  for (int p = 0; p < s.points; ++p) {
    std::size_t k = 0;
    for (auto i = static_cast<std::size_t>(s.pt_ptr[static_cast<std::size_t>(p)]);
         i < static_cast<std::size_t>(s.pt_ptr[static_cast<std::size_t>(p) + 1]); ++i) {
      k += static_cast<std::size_t>(
          cam_free[static_cast<std::size_t>(s.obs_cam[static_cast<std::size_t>(s.pt_obs[i])])]);
    }
    n_terms += k * k;
  }
  const std::size_t bytes = n_terms * 3 * sizeof(int);
  const std::size_t limit = std::min<std::size_t>(
      host_memory_bytes() / 2,
      static_cast<std::size_t>(std::numeric_limits<int>::max()) * 3 * sizeof(int));
  if (bytes > limit) {
    throw std::runtime_error("ba cuda: " + std::to_string(n_terms) +
                             " Schur terms (sum over points of views^2) need " +
                             std::to_string(bytes >> 20) + " MiB, over the limit of " +
                             std::to_string(limit >> 20) + " MiB");
  }
  // Terms per pair of free cameras (a, b), dense over the free cameras only (17 bytes per entry:
  // 24 MB at 1200 free cameras; a local BA has a few dozen); pairs in (a, b) order, every free
  // camera's diagonal block included, terms in point order as before.
  std::vector<int> free_index(nc, -1);
  std::vector<int> free_cams;
  for (std::size_t c = 0; c < nc; ++c) {
    if (cam_free[c] != 0) {
      free_index[c] = static_cast<int>(free_cams.size());
      free_cams.push_back(static_cast<int>(c));
    }
  }
  const std::size_t nf = free_cams.size();
  std::vector<std::size_t> count(nf * nf, 0);
  std::vector<char> used(nf * nf, 0);
  for (std::size_t c = 0; c < nf; ++c) used[c * nf + c] = 1;
  const auto each_term = [&](auto&& f) {
    for (int p = 0; p < s.points; ++p) {
      const auto lo = static_cast<std::size_t>(s.pt_ptr[static_cast<std::size_t>(p)]);
      const auto hi = static_cast<std::size_t>(s.pt_ptr[static_cast<std::size_t>(p) + 1]);
      for (std::size_t i = lo; i < hi; ++i) {
        const int oa = s.pt_obs[i];
        const int fa =
            free_index[static_cast<std::size_t>(s.obs_cam[static_cast<std::size_t>(oa)])];
        if (fa < 0) continue;
        for (std::size_t j = lo; j < hi; ++j) {
          const int ob = s.pt_obs[j];
          const int fb =
              free_index[static_cast<std::size_t>(s.obs_cam[static_cast<std::size_t>(ob)])];
          if (fb < 0) continue;
          f(static_cast<std::size_t>(fa) * nf + static_cast<std::size_t>(fb), p, oa, ob);
        }
      }
    }
  };
  each_term([&](std::size_t key, int, int, int) {
    ++count[key];
    used[key] = 1;
  });
  s.pair_a.clear();
  s.pair_b.clear();
  s.term_ptr.assign(1, 0);
  std::vector<std::size_t> cursor(nf * nf, 0);
  for (std::size_t key = 0; key < nf * nf; ++key) {
    if (used[key] == 0) continue;
    s.pair_a.push_back(free_cams[key / nf]);
    s.pair_b.push_back(free_cams[key % nf]);
    cursor[key] = static_cast<std::size_t>(s.term_ptr.back());
    s.term_ptr.push_back(static_cast<int>(cursor[key] + count[key]));
  }
  count = {};
  used = {};
  s.term_oa.assign(n_terms, 0);
  s.term_ob.assign(n_terms, 0);
  each_term([&](std::size_t key, int, int oa, int ob) {
    const std::size_t at = cursor[key]++;
    s.term_oa[at] = oa;
    s.term_ob[at] = ob;
  });
}

namespace {

// Above this many free unknowns (6 per free camera) the reduced system goes to cuSOLVER's
// blocked Cholesky, which uses every SM; below it the one-block kernel wins on launch overhead.
// P9a, bench_ba --gpu-only, ms per LM iteration (one-block / cuSOLVER): 12 cameras 0.37 / 0.42,
// 36: 2.95 / 2.49, 144: 127 / 107, 576: 1895 / 84. The one-block kernel also needs a column of L
// in shared memory (40 KB: at most 5120 unknowns).
constexpr int kOneBlockMaxFree = 96;  // 16 free cameras
static_assert(kOneBlockMaxFree * sizeof(double) <= 40 * 1024, "one column of L in shared memory");

void check_solver(cusolverStatus_t s, const char* what) {
  if (s != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("ba cuda: ") + what + " failed (cusolver status " +
                             std::to_string(static_cast<int>(s)) + ")");
  }
}

// cuSOLVER dense Cholesky for large reduced systems; created on first use.
struct DenseSolver {
  cusolverDnHandle_t handle = nullptr;
  core::DeviceBuffer<double> work;
  core::DeviceBuffer<int> info;
  int n = -1;
  DenseSolver() = default;
  DenseSolver(const DenseSolver&) = delete;
  DenseSolver& operator=(const DenseSolver&) = delete;
  ~DenseSolver() {
    if (handle != nullptr) cusolverDnDestroy(handle);  // no throw in cleanup
  }
  // Factors S (n x n, row-major lower triangle = column-major upper) and solves S x = b in place;
  // writes 1 to *ok if S is positive definite, else 0.
  void solve(double* S, int n_, double* b, int* ok, cudaStream_t st) {
    if (handle == nullptr) {
      check_solver(cusolverDnCreate(&handle), "cusolverDnCreate");
      info = core::DeviceBuffer<int>(1);
    }
    check_solver(cusolverDnSetStream(handle, st), "cusolverDnSetStream");
    if (n_ != n) {
      int lwork = 0;
      check_solver(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, n_, S, n_, &lwork),
                   "potrf_bufferSize");
      work = core::DeviceBuffer<double>(static_cast<std::size_t>(std::max(lwork, 1)));
      n = n_;
    }
    check_solver(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_UPPER, n_, S, n_, work.data(),
                                  static_cast<int>(work.size()), info.data()),
                 "potrf");
    int h_info = 0;
    CHECK(cudaMemcpyAsync(&h_info, info.data(), sizeof(int), cudaMemcpyDeviceToHost, st));
    CHECK(cudaStreamSynchronize(st));
    const int good = h_info == 0 ? 1 : 0;
    if (good == 1) {
      check_solver(
          cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_UPPER, n_, 1, S, n_, b, n_, info.data()),
          "potrs");
    }
    CHECK(cudaMemcpyAsync(ok, &good, sizeof(int), cudaMemcpyHostToDevice, st));
    CHECK(cudaStreamSynchronize(st));
  }
};

}  // namespace

struct Device::Impl {
  Structure s;
  DenseSolver dense;
  core::Stream stream;
  int nobs = 0;
  core::DeviceBuffer<int> obs_cam, obs_pt, obs_sensor, cam_ptr, cam_obs, pt_ptr, pt_obs, free_col;
  core::DeviceBuffer<int> pair_a, pair_b, term_ptr, term_oa, term_ob;
  int n_pairs = 0;
  core::DeviceBuffer<double> uv, K, sensor, R, t, X;
  core::DeviceBuffer<double> Jc, Jx, r, w, rho, partial;
  core::DeviceBuffer<double> U, V, W, WV, gc, gp, Vinv, y, S, b, dp, cam_partial;
  core::DeviceBuffer<int> ok_pt, ok;
};

Device::Device(Structure structure) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.s = std::move(structure);
  const Structure& s = m.s;
  m.nobs = static_cast<int>(s.obs_cam.size());
  m.n_pairs = static_cast<int>(s.pair_a.size());
  const core::Stream& st = m.stream;
  m.obs_cam = make(s.obs_cam, st);
  m.obs_pt = make(s.obs_pt, st);
  m.obs_sensor = make(s.obs_sensor, st);
  m.sensor = make(s.sensor, st);
  m.cam_ptr = make(s.cam_ptr, st);
  m.cam_obs = make(s.cam_obs, st);
  m.pt_ptr = make(s.pt_ptr, st);
  m.pt_obs = make(s.pt_obs, st);
  m.free_col = make(s.free_col, st);
  m.pair_a = make(s.pair_a, st);
  m.pair_b = make(s.pair_b, st);
  m.term_ptr = make(s.term_ptr, st);
  m.term_oa = make(s.term_oa, st);
  m.term_ob = make(s.term_ob, st);
  m.uv = make(s.uv, st);
  m.K = make(s.K, st);
  const auto nc = static_cast<std::size_t>(s.cameras);
  const auto np = static_cast<std::size_t>(s.points);
  const auto no = static_cast<std::size_t>(std::max(m.nobs, 1));
  m.R = core::DeviceBuffer<double>(9 * std::max<std::size_t>(nc, 1));
  m.t = core::DeviceBuffer<double>(3 * std::max<std::size_t>(nc, 1));
  m.X = core::DeviceBuffer<double>(3 * std::max<std::size_t>(np, 1));
  m.Jc = core::DeviceBuffer<double>(12 * no);
  m.Jx = core::DeviceBuffer<double>(6 * no);
  m.r = core::DeviceBuffer<double>(2 * no);
  m.w = core::DeviceBuffer<double>(no);
  m.rho = core::DeviceBuffer<double>(no);
  m.partial = core::DeviceBuffer<double>(static_cast<std::size_t>(blocks(m.nobs) + 1));
  m.U = core::DeviceBuffer<double>(36 * std::max<std::size_t>(nc, 1));
  m.cam_partial = core::DeviceBuffer<double>(42 * kChunks * std::max<std::size_t>(nc, 1));
  m.gc = core::DeviceBuffer<double>(6 * std::max<std::size_t>(nc, 1));
  m.V = core::DeviceBuffer<double>(9 * std::max<std::size_t>(np, 1));
  m.gp = core::DeviceBuffer<double>(3 * std::max<std::size_t>(np, 1));
  m.W = core::DeviceBuffer<double>(18 * no);
  m.WV = core::DeviceBuffer<double>(18 * no);
  m.Vinv = core::DeviceBuffer<double>(9 * std::max<std::size_t>(np, 1));
  m.y = core::DeviceBuffer<double>(3 * std::max<std::size_t>(np, 1));
  m.dp = core::DeviceBuffer<double>(3 * std::max<std::size_t>(np, 1));
  const auto nf = static_cast<std::size_t>(std::max(s.n_free, 1));
  m.S = core::DeviceBuffer<double>(nf * nf);
  m.b = core::DeviceBuffer<double>(nf);
  m.ok_pt = core::DeviceBuffer<int>(std::max<std::size_t>(np, 1));
  m.ok = core::DeviceBuffer<int>(1);
  st.synchronize();
  // Only the counts and free_col are read from the host copy after this.
  for (auto* v :
       {&m.s.obs_cam, &m.s.obs_pt, &m.s.obs_sensor, &m.s.cam_ptr, &m.s.cam_obs, &m.s.pt_ptr,
        &m.s.pt_obs, &m.s.pair_a, &m.s.pair_b, &m.s.term_ptr, &m.s.term_oa, &m.s.term_ob}) {
    std::vector<int>().swap(*v);
  }
  std::vector<double>().swap(m.s.uv);
  std::vector<double>().swap(m.s.K);
  std::vector<double>().swap(m.s.sensor);
}

int Device::observations() const { return impl_->nobs; }

Device::~Device() = default;

void Device::set_params(const std::vector<double>& R, const std::vector<double>& t,
                        const std::vector<double>& X) {
  Impl& m = *impl_;
  m.R.upload(R.data(), R.size(), m.stream);
  m.t.upload(t.data(), t.size(), m.stream);
  if (!X.empty()) m.X.upload(X.data(), X.size(), m.stream);
}

double Device::cost(double huber) {
  Impl& m = *impl_;
  if (m.nobs == 0) return 0.0;
  const cudaStream_t st = m.stream.get();
  obs_kernel<<<blocks(m.nobs), kBlock, 0, st>>>(
      m.nobs, m.obs_cam.data(), m.obs_pt.data(), m.obs_sensor.data(), m.uv.data(), m.K.data(),
      m.sensor.data(), m.R.data(), m.t.data(), m.X.data(), huber, m.Jc.data(), m.Jx.data(),
      m.r.data(), m.w.data(), m.rho.data());
  sum_kernel<<<blocks(m.nobs), kBlock, 0, st>>>(m.rho.data(), m.nobs, m.partial.data());
  CHECK(cudaGetLastError());
  std::vector<double> part(static_cast<std::size_t>(blocks(m.nobs)));
  m.partial.download(part.data(), part.size(), m.stream);
  m.stream.synchronize();
  double sum = 0.0;
  for (const double v : part) sum += v;
  return 0.5 * sum;
}

void Device::linearize(double huber) {
  Impl& m = *impl_;
  if (m.nobs == 0) return;
  const cudaStream_t st = m.stream.get();
  obs_kernel<<<blocks(m.nobs), kBlock, 0, st>>>(
      m.nobs, m.obs_cam.data(), m.obs_pt.data(), m.obs_sensor.data(), m.uv.data(), m.K.data(),
      m.sensor.data(), m.R.data(), m.t.data(), m.X.data(), huber, m.Jc.data(), m.Jx.data(),
      m.r.data(), m.w.data(), m.rho.data());
  cam_kernel<<<dim3(static_cast<unsigned int>(m.s.cameras), kChunks), 64, 0, st>>>(
      m.cam_ptr.data(), m.cam_obs.data(), m.Jc.data(), m.r.data(), m.w.data(),
      m.cam_partial.data());
  cam_sum_kernel<<<blocks(42 * m.s.cameras), kBlock, 0, st>>>(m.s.cameras, m.cam_partial.data(),
                                                              m.U.data(), m.gc.data());
  pt_kernel<<<blocks(m.s.points), kBlock, 0, st>>>(m.s.points, m.pt_ptr.data(), m.pt_obs.data(),
                                                   m.Jx.data(), m.r.data(), m.w.data(), m.V.data(),
                                                   m.gp.data());
  w_kernel<<<blocks(m.nobs), kBlock, 0, st>>>(m.nobs, m.Jc.data(), m.Jx.data(), m.w.data(),
                                              m.W.data());
  CHECK(cudaGetLastError());
}

void Device::download(std::vector<double>* U, std::vector<double>* V, std::vector<double>* W,
                      std::vector<double>* gc, std::vector<double>* gp) const {
  Impl& m = *impl_;
  const auto get = [&](const core::DeviceBuffer<double>& d, std::vector<double>* h, std::size_t n) {
    if (h == nullptr) return;
    h->resize(n);
    if (n > 0) d.download(h->data(), n, m.stream);
  };
  get(m.U, U, 36 * static_cast<std::size_t>(m.s.cameras));
  get(m.V, V, 9 * static_cast<std::size_t>(m.s.points));
  get(m.W, W, 18 * static_cast<std::size_t>(m.nobs));
  get(m.gc, gc, 6 * static_cast<std::size_t>(m.s.cameras));
  get(m.gp, gp, 3 * static_cast<std::size_t>(m.s.points));
  m.stream.synchronize();
}

bool Device::solve(double lambda, std::vector<double>& dc, std::vector<double>& dp) {
  Impl& m = *impl_;
  const cudaStream_t st = m.stream.get();
  const int nf = m.s.n_free;
  dc.assign(6 * static_cast<std::size_t>(m.s.cameras), 0.0);
  dp.assign(3 * static_cast<std::size_t>(m.s.points), 0.0);
  if (m.s.points > 0) {
    vinv_kernel<<<blocks(m.s.points), kBlock, 0, st>>>(m.s.points, m.V.data(), m.gp.data(), lambda,
                                                       m.Vinv.data(), m.y.data(), m.ok_pt.data());
  }
  CHECK(cudaMemsetAsync(m.S.data(), 0, m.S.bytes(), st));
  CHECK(cudaMemsetAsync(m.b.data(), 0, m.b.bytes(), st));
  b_kernel<<<dim3(static_cast<unsigned int>(m.s.cameras), kChunks), 32, 0, st>>>(
      m.cam_ptr.data(), m.cam_obs.data(), m.obs_pt.data(), m.W.data(), m.y.data(),
      m.cam_partial.data());
  b_sum_kernel<<<blocks(6 * m.s.cameras), kBlock, 0, st>>>(
      m.s.cameras, m.cam_partial.data(), m.gc.data(), m.free_col.data(), m.b.data());
  wv_kernel<<<blocks(m.nobs), kBlock, 0, st>>>(m.nobs, m.obs_pt.data(), m.W.data(), m.Vinv.data(),
                                               m.WV.data());
  if (m.n_pairs > 0) {  // none when every camera is fixed
    s_kernel<<<static_cast<unsigned int>(m.n_pairs), 64, 0, st>>>(
        m.pair_a.data(), m.pair_b.data(), m.term_ptr.data(), m.term_oa.data(), m.term_ob.data(),
        m.U.data(), m.W.data(), m.WV.data(), lambda, m.free_col.data(), nf, m.S.data());
  }
  if (nf > kOneBlockMaxFree) {
    CHECK(cudaGetLastError());
    m.dense.solve(m.S.data(), nf, m.b.data(), m.ok.data(), st);
  } else if (nf > 0) {
    cholesky_solve_kernel<<<1, 1024, static_cast<std::size_t>(nf) * sizeof(double), st>>>(
        m.S.data(), nf, m.b.data(), m.ok.data());
  }
  CHECK(cudaGetLastError());
  int ok = 1;
  if (nf > 0) m.ok.download(&ok, 1, m.stream);
  m.stream.synchronize();
  if (ok == 0) return false;
  if (m.s.points > 0) {
    dp_kernel<<<blocks(m.s.points), kBlock, 0, st>>>(
        m.s.points, m.pt_ptr.data(), m.pt_obs.data(), m.obs_cam.data(), m.W.data(), m.Vinv.data(),
        m.gp.data(), m.b.data(), m.free_col.data(), m.dp.data());
    CHECK(cudaGetLastError());
    m.dp.download(dp.data(), dp.size(), m.stream);
  }
  std::vector<double> x(static_cast<std::size_t>(std::max(nf, 1)));
  if (nf > 0) m.b.download(x.data(), static_cast<std::size_t>(nf), m.stream);
  m.stream.synchronize();
  for (std::size_t c = 0; c < static_cast<std::size_t>(m.s.cameras); ++c) {
    for (std::size_t a = 0; a < 6; ++a) {
      const int col = m.s.free_col[6 * c + a];
      if (col >= 0) dc[6 * c + a] = x[static_cast<std::size_t>(col)];
    }
  }
  for (const double v : dc) {
    if (!std::isfinite(v)) return false;
  }
  return true;
}

}  // namespace sfm::ba::cuda
