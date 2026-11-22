// knn2_cuda: packs descriptors and pairs for the kernel (kernels.cuh) and unpacks the result.
#include <algorithm>
#include <stdexcept>

#include "sfm/matching/bf_matcher.hpp"
#include "sfm/matching/matching_cuda.hpp"

#include "kernels.cuh"

namespace sfm::matching {

std::vector<std::vector<Knn2>> knn2_cuda(const std::vector<features::Features>& features,
                                         const std::vector<std::pair<int, int>>& pairs) {
  for (const auto& f : features) {
    if (f.words != features.front().words) {
      throw std::invalid_argument("knn2_cuda: descriptor widths differ between images");
    }
  }
  const int words = features.empty() ? features::kDescriptorWords : features.front().words;
  if (words != features::kDescriptorWords && words != features::kSiftWords) {
    throw std::invalid_argument("knn2_cuda: unsupported descriptor width");
  }
  std::vector<int> desc_offset(features.size() + 1, 0);
  for (std::size_t i = 0; i < features.size(); ++i) {
    desc_offset[i + 1] = desc_offset[i] + static_cast<int>(features[i].size());
  }
  std::vector<unsigned int> desc;
  desc.reserve(static_cast<std::size_t>(words) * static_cast<std::size_t>(desc_offset.back()));
  for (const auto& f : features) desc.insert(desc.end(), f.desc.begin(), f.desc.end());
  std::vector<cuda::KnnPair> kp;
  kp.reserve(pairs.size());
  int out_total = 0;
  int max_query = 0;
  for (const auto& [i, j] : pairs) {
    const auto a = static_cast<std::size_t>(i);
    const auto b = static_cast<std::size_t>(j);
    const int nq = static_cast<int>(features[a].size());
    kp.push_back(
        {desc_offset[a], nq, desc_offset[b], static_cast<int>(features[b].size()), out_total});
    out_total += nq;
    max_query = std::max(max_query, nq);
  }
  const std::vector<int> raw = cuda::knn2(desc, kp, out_total, max_query, words);
  std::vector<std::vector<Knn2>> out(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    out[k].resize(static_cast<std::size_t>(kp[k].n_query));
    for (int q = 0; q < kp[k].n_query; ++q) {
      const auto o = 4 * static_cast<std::size_t>(kp[k].out_offset + q);
      out[k][static_cast<std::size_t>(q)] = {raw[o], raw[o + 1], raw[o + 2], raw[o + 3]};
    }
  }
  return out;
}

}  // namespace sfm::matching
