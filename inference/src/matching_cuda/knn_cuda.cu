// Batched brute-force top-2 on the GPU: Hamming (ORB, 8 words; one thread per (pair, query), train
// descriptors through shared memory in tiles) or squared L2 (SIFT, 32 words of 4 bytes; dot
// products on the tensor cores). Equal to knn2: distances are exact integers and each query's
// top-2 is the two smallest (distance, train index) pairs, which is what the CPU's scan in
// increasing index with strict comparisons keeps.
#include <algorithm>
#include <cstdint>
#include <vector>

#include <mma.h>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/stream.hpp"

#include "kernels.cuh"

namespace sfm::matching::cuda {

namespace {

constexpr int kTile = 128;  // threads per block = queries per block = train tile size

__global__ void knn2_kernel(const unsigned int* d_desc, const KnnPair* d_pairs, int4* d_out) {
  __shared__ unsigned int tile[kTile][8];
  const KnnPair pd = d_pairs[blockIdx.y];
  const int q = blockIdx.x * kTile + threadIdx.x;
  if (blockIdx.x * kTile >= pd.n_query) return;  // whole block idle (uniform)
  const bool active = q < pd.n_query;
  unsigned int mine[8];
  for (int w = 0; w < 8; ++w) mine[w] = active ? d_desc[8LL * (pd.query_offset + q) + w] : 0U;
  int b1 = -1;
  int d1 = 257;
  int b2 = -1;
  int d2 = 257;
  for (int base = 0; base < pd.n_train; base += kTile) {
    const int t_load = base + threadIdx.x;
    for (int w = 0; w < 8; ++w) {
      tile[threadIdx.x][w] =
          t_load < pd.n_train ? d_desc[8LL * (pd.train_offset + t_load) + w] : 0U;
    }
    __syncthreads();
    const int n = min(kTile, pd.n_train - base);
    for (int t = 0; t < n; ++t) {
      int d = 0;
      for (int w = 0; w < 8; ++w) d += __popc(mine[w] ^ tile[t][w]);
      if (d < d1) {
        b2 = b1;
        d2 = d1;
        b1 = base + t;
        d1 = d;
      } else if (d < d2) {
        b2 = base + t;
        d2 = d;
      }
    }
    __syncthreads();
  }
  if (active) d_out[pd.out_offset + q] = make_int4(b1, d1, b2, d2);
}

constexpr int kNoL2 = 1 << 30;  // = matching::kNoL2

// |d|^2 of every descriptor (exact in int32: <= 128 * 255^2).
__global__ void sq_norm_kernel(const unsigned int* d_desc, int n, int* d_norm) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int acc = 0;
  for (int w = 0; w < 32; ++w) {
    const unsigned int v = d_desc[32LL * i + w];
    acc = __dp4a(v, v, acc);
  }
  d_norm[i] = static_cast<int>(acc);
}

// Top-2 entries (distance, index) ordered lexicographically: the CPU's scan in increasing index
// with strict comparisons keeps exactly the two smallest (d, index) pairs.
struct Top2 {
  int b1 = -1;
  int d1 = kNoL2;
  int b2 = -1;
  int d2 = kNoL2;
};

__device__ bool less_entry(int da, int ba, int db, int bb) {
  return da < db || (da == db && static_cast<unsigned int>(ba) < static_cast<unsigned int>(bb));
}

__device__ void push(Top2& t, int d, int b) {  // b larger than every index seen so far
  if (d < t.d1) {
    t.b2 = t.b1;
    t.d2 = t.d1;
    t.b1 = b;
    t.d1 = d;
  } else if (d < t.d2) {
    t.b2 = b;
    t.d2 = d;
  }
}

__device__ Top2 merge(const Top2& a, const Top2& c) {  // disjoint index sets
  Top2 r;
  if (less_entry(c.d1, c.b1, a.d1, a.b1)) {
    r.b1 = c.b1;
    r.d1 = c.d1;
    const bool a_next = less_entry(a.d1, a.b1, c.d2, c.b2);
    r.b2 = a_next ? a.b1 : c.b2;
    r.d2 = a_next ? a.d1 : c.d2;
  } else {
    r.b1 = a.b1;
    r.d1 = a.d1;
    const bool c_next = less_entry(c.d1, c.b1, a.d2, a.b2);
    r.b2 = c_next ? c.b1 : a.b2;
    r.d2 = c_next ? c.d1 : a.d2;
  }
  return r;
}

// SIFT top-2 on the tensor cores: a block takes Q queries of one pair and walks the train
// descriptors T at a time. The u8 dot products of the Q x T tile come from WMMA (16x16x16,
// u8 x u8 -> s32, exact), one 16-query strip per warp; then 2 threads per query scan the tile
// (even / odd columns) and keep their top-2, merged at the end. Integer-exact, so equal to the
// CPU.
template <int Q, int T>
struct WmmaTile {
  static constexpr int kThreads = 2 * Q;  // = 32 * (Q / 16) warps
  static constexpr int kLdc = T + 4;      // int32 row stride of the dot tile (multiple of 4)
};

template <int Q, int T>
__global__ void __launch_bounds__(2 * Q)
    knn2_l2_wmma_kernel(const unsigned int* d_desc, const int* d_norm, const KnnPair* d_pairs,
                        int4* d_out) {
  constexpr int kLdc = WmmaTile<Q, T>::kLdc;
  // Descriptors as 8 panels of 16 bytes (k = 16 kk .. 16 kk + 15) so that every fragment pointer
  // is 32-byte aligned with a 16-byte leading dimension.
  __shared__ __align__(32) unsigned int q_panel[8][Q][4];
  __shared__ __align__(32) unsigned int t_panel[8][T][4];
  __shared__ __align__(32) int dot[Q][kLdc];
  __shared__ int t_norm[T];
  const KnnPair pd = d_pairs[blockIdx.y];
  const int q0 = blockIdx.x * Q;
  if (q0 >= pd.n_query) return;  // whole block idle (uniform)
  const int tid = threadIdx.x;
  const int warp = tid / 32;

  for (int k = tid; k < Q * 32; k += 2 * Q) {  // coalesced: word k % 32 of row k / 32
    const int row = k / 32;
    const int w = k % 32;
    const int q = q0 + row;
    q_panel[w / 4][row][w % 4] = q < pd.n_query ? d_desc[32LL * (pd.query_offset + q) + w] : 0U;
  }

  const int my_row = tid / 2;  // scan: 2 threads per query
  const int half = tid % 2;
  const int my_q = q0 + my_row;
  const int my_norm = my_q < pd.n_query ? d_norm[pd.query_offset + my_q] : 0;
  Top2 best;

  for (int base = 0; base < pd.n_train; base += T) {
    for (int k = tid; k < T * 32; k += 2 * Q) {
      const int row = k / 32;
      const int w = k % 32;
      const int t = base + row;
      t_panel[w / 4][row][w % 4] = t < pd.n_train ? d_desc[32LL * (pd.train_offset + t) + w] : 0U;
    }
    for (int k = tid; k < T; k += 2 * Q) {
      const int t = base + k;
      t_norm[k] = t < pd.n_train ? d_norm[pd.train_offset + t] : 0;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> acc[T / 16];
    for (int c = 0; c < T / 16; ++c) nvcuda::wmma::fill_fragment(acc[c], 0);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, unsigned char,
                             nvcuda::wmma::row_major>
          a;
      nvcuda::wmma::load_matrix_sync(
          a, reinterpret_cast<const unsigned char*>(&q_panel[kk][16 * warp][0]), 16);
#pragma unroll
      for (int c = 0; c < T / 16; ++c) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, unsigned char,
                               nvcuda::wmma::col_major>
            b;
        nvcuda::wmma::load_matrix_sync(
            b, reinterpret_cast<const unsigned char*>(&t_panel[kk][16 * c][0]), 16);
        nvcuda::wmma::mma_sync(acc[c], a, b, acc[c]);
      }
    }
    for (int c = 0; c < T / 16; ++c) {
      nvcuda::wmma::store_matrix_sync(&dot[16 * warp][16 * c], acc[c], kLdc,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    const int n = min(T, pd.n_train - base);
    for (int t = half; t < n; t += 2) {
      push(best, my_norm + t_norm[t] - 2 * dot[my_row][t], base + t);
    }
    __syncthreads();  // the tiles are overwritten next
  }

  Top2 other;
  other.b1 = __shfl_xor_sync(0xffffffffU, best.b1, 1);
  other.d1 = __shfl_xor_sync(0xffffffffU, best.d1, 1);
  other.b2 = __shfl_xor_sync(0xffffffffU, best.b2, 1);
  other.d2 = __shfl_xor_sync(0xffffffffU, best.d2, 1);
  const Top2 r = merge(best, other);
  if (half == 0 && my_q < pd.n_query) {
    d_out[pd.out_offset + my_q] = make_int4(r.b1, r.d1, r.b2, r.d2);
  }
}

// 64 queries x 32 train: fastest of 16..256 x 16..64 on courtyard (RTX 5060 Ti, 703 pairs of
// 8.7k SIFT each way: 369 ms vs 700 ms for one dp4a thread per query).
constexpr int kQ = 64;
constexpr int kT = 32;

}  // namespace

std::vector<int> knn2(const std::vector<unsigned int>& desc, const std::vector<KnnPair>& pairs,
                      int out_total, int max_query, int words) {
  std::vector<int> h_out(4 * static_cast<std::size_t>(out_total));
  if (pairs.empty() || out_total == 0) return h_out;
  core::Stream stream;
  core::DeviceBuffer<unsigned int> d_desc(std::max<std::size_t>(desc.size(), 1));
  core::DeviceBuffer<KnnPair> d_pairs(pairs.size());
  core::DeviceBuffer<int4> d_out(static_cast<std::size_t>(out_total));
  if (!desc.empty()) d_desc.upload(desc.data(), desc.size(), stream);
  d_pairs.upload(pairs.data(), pairs.size(), stream);
  // Squared norms once per descriptor (SIFT only), shared by every pair.
  const int n_desc = words == 32 ? static_cast<int>(desc.size() / 32) : 0;
  core::DeviceBuffer<int> d_norm(static_cast<std::size_t>(std::max(n_desc, 1)));
  if (n_desc > 0) {
    sq_norm_kernel<<<static_cast<unsigned int>((n_desc + 255) / 256), 256, 0, stream.get()>>>(
        d_desc.data(), n_desc, d_norm.data());
    CHECK(cudaGetLastError());
  }
  for (std::size_t first = 0; first < pairs.size(); first += 65535) {  // grid y <= 65535
    const auto count =
        static_cast<unsigned int>(std::min<std::size_t>(65535, pairs.size() - first));
    if (words == 32) {
      const dim3 grid(static_cast<unsigned int>((max_query + kQ - 1) / kQ), count);
      knn2_l2_wmma_kernel<kQ, kT><<<grid, WmmaTile<kQ, kT>::kThreads, 0, stream.get()>>>(
          d_desc.data(), d_norm.data(), d_pairs.data() + first, d_out.data());
    } else {
      const dim3 grid(static_cast<unsigned int>((max_query + kTile - 1) / kTile), count);
      knn2_kernel<<<grid, kTile, 0, stream.get()>>>(d_desc.data(), d_pairs.data() + first,
                                                    d_out.data());
    }
    CHECK(cudaGetLastError());
  }
  CHECK(cudaMemcpyAsync(h_out.data(), d_out.data(), h_out.size() * sizeof(int),
                        cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  return h_out;
}

}  // namespace sfm::matching::cuda
