// Batched F-RANSAC on the GPU: one thread per (problem, hypothesis). Each thread draws 8 distinct
// correspondences (counter-based splitmix64, so hypothesis h of a problem is the same on every
// run), fits F with the normalised 8-point method (null vector by Gaussian elimination with full
// pivoting, rank 2 by removing the smallest right singular direction) and scores it over all
// correspondences of its problem (Sampson^2 < thr^2). A second kernel picks each problem's best
// hypothesis; the host refits it (refit_fundamental, shared with the CPU RANSAC).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"

#include "kernels.cuh"

namespace sfm::matching::cuda {

namespace {

constexpr int kBlock = 128;

__device__ unsigned long long splitmix64(unsigned long long& state) {
  unsigned long long z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Hartley normalisation of 8 points: returns s, cx, cy (x_n = s (x - c)); false if degenerate.
__device__ bool normalise8(const double* x, const double* y, double& s, double& cx, double& cy) {
  cx = 0;
  cy = 0;
  for (int k = 0; k < 8; ++k) {
    cx += x[k];
    cy += y[k];
  }
  cx /= 8;
  cy /= 8;
  double d = 0;
  for (int k = 0; k < 8; ++k) d += sqrt((x[k] - cx) * (x[k] - cx) + (y[k] - cy) * (y[k] - cy));
  d /= 8;
  if (!(d > 1e-12)) return false;
  s = sqrt(2.0) / d;
  return true;
}

// Null vector f (9) of the 8 x 9 matrix A (row-major, destroyed). False if rank < 8.
__device__ bool null_vector(double A[8][9], double f[9]) {
  int col_of[9];
  for (int c = 0; c < 9; ++c) col_of[c] = c;
  double scale = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 9; ++c) scale = fmax(scale, fabs(A[r][c]));
  }
  for (int k = 0; k < 8; ++k) {
    // Full pivot over the remaining rows and columns.
    int pr = k;
    int pc = k;
    double best = 0;
    for (int r = k; r < 8; ++r) {
      for (int c = k; c < 9; ++c) {
        if (fabs(A[r][c]) > best) {
          best = fabs(A[r][c]);
          pr = r;
          pc = c;
        }
      }
    }
    if (!(best > 1e-12 * scale)) return false;
    if (pr != k) {
      for (int c = 0; c < 9; ++c) {
        const double t = A[k][c];
        A[k][c] = A[pr][c];
        A[pr][c] = t;
      }
    }
    if (pc != k) {
      for (int r = 0; r < 8; ++r) {
        const double t = A[r][k];
        A[r][k] = A[r][pc];
        A[r][pc] = t;
      }
      const int t = col_of[k];
      col_of[k] = col_of[pc];
      col_of[pc] = t;
    }
    for (int r = k + 1; r < 8; ++r) {
      const double m = A[r][k] / A[k][k];
      for (int c = k; c < 9; ++c) A[r][c] -= m * A[k][c];
    }
  }
  // Upper-triangular 8 x 8 system with the free (permuted) column 8 set to 1.
  double z[9];
  z[8] = 1.0;
  for (int r = 7; r >= 0; --r) {
    double acc = A[r][8];
    for (int c = r + 1; c < 8; ++c) acc += A[r][c] * z[c];
    z[r] = -acc / A[r][r];
  }
  double norm = 0;
  for (int c = 0; c < 9; ++c) norm += z[c] * z[c];
  norm = sqrt(norm);
  for (int c = 0; c < 9; ++c) f[col_of[c]] = z[c] / norm;
  return true;
}

// Eigenvector of the smallest eigenvalue of the symmetric 3x3 M (cyclic Jacobi).
__device__ void smallest_eigvec3(double M[3][3], double v[3]) {
  double V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    const double off = M[0][1] * M[0][1] + M[0][2] * M[0][2] + M[1][2] * M[1][2];
    if (off < 1e-30) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (fabs(M[p][q]) < 1e-300) continue;
        const double theta = (M[q][q] - M[p][p]) / (2 * M[p][q]);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1));
        const double c = 1 / sqrt(t * t + 1);
        const double s = t * c;
        for (int k = 0; k < 3; ++k) {  // M <- J^T M J
          const double mkp = M[k][p];
          const double mkq = M[k][q];
          M[k][p] = c * mkp - s * mkq;
          M[k][q] = s * mkp + c * mkq;
        }
        for (int k = 0; k < 3; ++k) {
          const double mpk = M[p][k];
          const double mqk = M[q][k];
          M[p][k] = c * mpk - s * mqk;
          M[q][k] = s * mpk + c * mqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = V[k][p];
          const double vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int m = 0;
  if (M[1][1] < M[m][m]) m = 1;
  if (M[2][2] < M[m][m]) m = 2;
  for (int k = 0; k < 3; ++k) v[k] = V[k][m];
}

__global__ void hypotheses_kernel(const double* d_x1, const double* d_x2,
                                  const RansacProblem* d_prob, int n_prob, int iters, double thr2,
                                  double* d_F, int* d_inl, double* d_err) {
  const long long tid = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (tid >= static_cast<long long>(n_prob) * iters) return;
  const int pi = static_cast<int>(tid / iters);
  const int h = static_cast<int>(tid % iters);
  const RansacProblem pr = d_prob[pi];
  const long long out = pr.hyp_offset + h;
  d_inl[out] = -1;
  d_err[out] = 0.0;
  if (pr.n < 8) return;

  // 8 distinct indices.
  unsigned long long state =
      pr.seed ^ (0xD1B54A32D192ED03ULL * static_cast<unsigned long long>(h + 1));
  int idx[8];
  for (int k = 0; k < 8; ++k) {
    for (;;) {
      const auto r = static_cast<int>((splitmix64(state) >> 11) * 0x1.0p-53 * pr.n);
      const int cand = r < pr.n ? r : pr.n - 1;
      bool dup = false;
      for (int m = 0; m < k; ++m) dup = dup || idx[m] == cand;
      if (!dup) {
        idx[k] = cand;
        break;
      }
    }
  }
  double ax[8];
  double ay[8];
  double bx[8];
  double by[8];
  for (int k = 0; k < 8; ++k) {
    const long long o = 2LL * (pr.offset + idx[k]);
    ax[k] = d_x1[o];
    ay[k] = d_x1[o + 1];
    bx[k] = d_x2[o];
    by[k] = d_x2[o + 1];
  }
  double s1;
  double c1x;
  double c1y;
  double s2;
  double c2x;
  double c2y;
  if (!normalise8(ax, ay, s1, c1x, c1y) || !normalise8(bx, by, s2, c2x, c2y)) return;
  double A[8][9];
  for (int k = 0; k < 8; ++k) {
    const double u1 = s1 * (ax[k] - c1x);
    const double v1 = s1 * (ay[k] - c1y);
    const double u2 = s2 * (bx[k] - c2x);
    const double v2 = s2 * (by[k] - c2y);
    const double row[9] = {u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0};
    for (int c = 0; c < 9; ++c) A[k][c] = row[c];
  }
  double f[9];
  if (!null_vector(A, f)) return;
  // Rank 2: Fn <- Fn (I - v v^T), v = right singular vector of the smallest singular value.
  double M[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      M[i][j] = f[i] * f[j] + f[3 + i] * f[3 + j] + f[6 + i] * f[6 + j];  // Fn^T Fn
    }
  }
  double v[3];
  smallest_eigvec3(M, v);
  double Fn[3][3];
  for (int r = 0; r < 3; ++r) {
    const double fv = f[3 * r] * v[0] + f[3 * r + 1] * v[1] + f[3 * r + 2] * v[2];
    for (int c = 0; c < 3; ++c) Fn[r][c] = f[3 * r + c] - fv * v[c];
  }
  // Denormalise: F = T2^T Fn T1, T = [[s, 0, -s cx], [0, s, -s cy], [0, 0, 1]].
  const double T1[3][3] = {{s1, 0, -s1 * c1x}, {0, s1, -s1 * c1y}, {0, 0, 1}};
  const double T2[3][3] = {{s2, 0, -s2 * c2x}, {0, s2, -s2 * c2y}, {0, 0, 1}};
  double G[3][3];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double acc = 0;
      for (int k = 0; k < 3; ++k) acc += Fn[r][k] * T1[k][c];
      G[r][c] = acc;
    }
  }
  double F[9];
  double norm = 0;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double acc = 0;
      for (int k = 0; k < 3; ++k) acc += T2[k][r] * G[k][c];
      F[3 * r + c] = acc;
      norm += acc * acc;
    }
  }
  norm = sqrt(norm);
  if (!(norm > 0)) return;
  for (double& e : F) e /= norm;

  int inl = 0;
  double err = 0;
  for (int k = 0; k < pr.n; ++k) {
    const long long o = 2LL * (pr.offset + k);
    const double x = d_x1[o];
    const double y = d_x1[o + 1];
    const double u = d_x2[o];
    const double w = d_x2[o + 1];
    const double fa0 = F[0] * x + F[1] * y + F[2];
    const double fa1 = F[3] * x + F[4] * y + F[5];
    const double fa2 = F[6] * x + F[7] * y + F[8];
    const double ftb0 = F[0] * u + F[3] * w + F[6];
    const double ftb1 = F[1] * u + F[4] * w + F[7];
    const double e = u * fa0 + w * fa1 + fa2;
    const double den = fa0 * fa0 + fa1 * fa1 + ftb0 * ftb0 + ftb1 * ftb1;
    const double d2 = den > 0 ? e * e / den : 0.0;
    if (d2 < thr2) {
      ++inl;
      err += d2;
    }
  }
  d_inl[out] = inl;
  d_err[out] = err;
  for (int c = 0; c < 9; ++c) d_F[9 * out + c] = F[c];
}

// One block per problem: best hypothesis (inliers desc, error asc, iteration asc).
__global__ void best_kernel(const RansacProblem* d_prob, int iters, const int* d_inl,
                            const double* d_err, int* d_best) {
  __shared__ int s_idx[kBlock];
  const RansacProblem pr = d_prob[blockIdx.x];
  int best = -1;
  for (int h = threadIdx.x; h < iters; h += blockDim.x) {
    const long long o = pr.hyp_offset + h;
    if (d_inl[o] < 0) continue;
    if (best < 0) {
      best = h;
      continue;
    }
    const long long b = pr.hyp_offset + best;
    if (d_inl[o] > d_inl[b] || (d_inl[o] == d_inl[b] && d_err[o] < d_err[b])) best = h;
  }
  s_idx[threadIdx.x] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const int a = s_idx[threadIdx.x];
      const int c = s_idx[threadIdx.x + stride];
      if (c >= 0) {
        bool take = a < 0;
        if (!take) {
          const long long oa = pr.hyp_offset + a;
          const long long oc = pr.hyp_offset + c;
          take = d_inl[oc] > d_inl[oa] ||
                 (d_inl[oc] == d_inl[oa] &&
                  (d_err[oc] < d_err[oa] || (d_err[oc] == d_err[oa] && c < a)));
        }
        if (take) s_idx[threadIdx.x] = c;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) d_best[blockIdx.x] = s_idx[0];
}

}  // namespace

std::vector<double> ransac_best(const std::vector<double>& x1, const std::vector<double>& x2,
                                const std::vector<RansacProblem>& problems, int iters,
                                double thr2) {
  const std::size_t n_prob = problems.size();
  std::vector<double> out(9 * n_prob, 0.0);
  if (n_prob == 0) return out;
  const std::size_t n_hyp = n_prob * static_cast<std::size_t>(iters);
  core::Stream stream;
  core::DeviceBuffer<double> d_x1(std::max<std::size_t>(x1.size(), 1));
  core::DeviceBuffer<double> d_x2(std::max<std::size_t>(x2.size(), 1));
  core::DeviceBuffer<RansacProblem> d_prob(n_prob);
  core::DeviceBuffer<double> d_F(9 * n_hyp);
  core::DeviceBuffer<int> d_inl(n_hyp);
  core::DeviceBuffer<double> d_err(n_hyp);
  core::DeviceBuffer<int> d_best(n_prob);
  if (!x1.empty()) {
    d_x1.upload(x1.data(), x1.size(), stream);
    d_x2.upload(x2.data(), x2.size(), stream);
  }
  d_prob.upload(problems.data(), n_prob, stream);
  const auto blocks = static_cast<unsigned int>((n_hyp + kBlock - 1) / kBlock);
  hypotheses_kernel<<<blocks, kBlock, 0, stream.get()>>>(d_x1.data(), d_x2.data(), d_prob.data(),
                                                         static_cast<int>(n_prob), iters, thr2,
                                                         d_F.data(), d_inl.data(), d_err.data());
  CHECK(cudaGetLastError());
  best_kernel<<<static_cast<unsigned int>(n_prob), kBlock, 0, stream.get()>>>(
      d_prob.data(), iters, d_inl.data(), d_err.data(), d_best.data());
  CHECK(cudaGetLastError());
  std::vector<int> best(n_prob);
  d_best.download(best.data(), n_prob, stream);
  stream.synchronize();
  for (std::size_t k = 0; k < n_prob; ++k) {
    if (best[k] < 0) continue;
    d_F.download(out.data() + 9 * k, 9, stream,
                 9 * (k * static_cast<std::size_t>(iters) + static_cast<std::size_t>(best[k])));
  }
  stream.synchronize();
  return out;
}

}  // namespace sfm::matching::cuda
