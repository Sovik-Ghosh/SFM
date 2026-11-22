// ransac_fundamental_cuda: packs the problems for the kernels (kernels.cuh), refits on the host.
#include <algorithm>

#include "sfm/matching/matching_cuda.hpp"

#include "kernels.cuh"

namespace sfm::matching {

std::vector<RansacResult> ransac_fundamental_cuda(const std::vector<Points2>& x1,
                                                  const std::vector<Points2>& x2,
                                                  const std::vector<std::uint64_t>& seeds,
                                                  const RansacParams& p) {
  const std::size_t n_prob = x1.size();
  const int iters = std::max(1, p.max_iters);
  std::vector<cuda::RansacProblem> prob(n_prob);
  std::vector<double> a;
  std::vector<double> b;
  int offset = 0;
  for (std::size_t k = 0; k < n_prob; ++k) {
    const int n = x2[k].size() == x1[k].size() ? static_cast<int>(x1[k].size()) : 0;
    prob[k] = {offset, n, seeds[k], static_cast<int>(k) * iters};
    for (std::size_t m = 0; m < static_cast<std::size_t>(n); ++m) {
      a.insert(a.end(), {x1[k][m].x(), x1[k][m].y()});
      b.insert(b.end(), {x2[k][m].x(), x2[k][m].y()});
    }
    offset += n;
  }
  const std::vector<double> best =
      cuda::ransac_best(a, b, prob, iters, p.threshold_px * p.threshold_px);
  std::vector<RansacResult> out(n_prob);
  for (std::size_t k = 0; k < n_prob; ++k) {
    Eigen::Matrix3d F;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) F(r, c) = best[9 * k + static_cast<std::size_t>(3 * r + c)];
    }
    out[k] = refit_fundamental(x1[k], x2[k], F, p.threshold_px);
    out[k].iterations = prob[k].n >= 8 ? iters : 0;
  }
  return out;
}

}  // namespace sfm::matching
