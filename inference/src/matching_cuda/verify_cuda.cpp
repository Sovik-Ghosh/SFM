// verify_all_pairs with top-2 and RANSAC on the GPU. The ratio / mutual test runs on the host on
// the GPU's top-2 (same comparisons as match_ratio, so the putative matches are identical).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "sfm/core/parallel.hpp"
#include "sfm/core/trace.hpp"
#include "sfm/matching/matching_cuda.hpp"

namespace sfm::matching {

std::vector<PairMatches> verify_all_pairs_cuda(const std::vector<features::Features>& features,
                                               const VerifyParams& p) {
  return verify_pairs_cuda(features,
                           select_pairs(static_cast<int>(features.size()), "exhaustive", 0, 0), p);
}

std::vector<PairMatches> verify_pairs_cuda(const std::vector<features::Features>& features,
                                           const std::vector<std::pair<int, int>>& pairs,
                                           const VerifyParams& p,
                                           std::size_t max_queries_per_chunk) {
  const int words = features.empty() ? features::kDescriptorWords : features.front().words;
  // RANSAC problems of one chunk of pairs. Each chunk owns its data, so a worker thread can verify
  // chunk c while this thread runs the GPU top-2 of chunk c + 1 (P9c: on the 4-camera
  // delivery_area the top-2 phase took 12.8 s and the CPU E-RANSAC 12.3 s, one after the other).
  struct Chunk {
    std::vector<std::size_t> kept;  // pair index of each RANSAC problem
    std::vector<std::vector<Match>> putative;
    std::vector<Points2> x1;
    std::vector<Points2> x2;
    std::vector<std::uint64_t> seeds;
    std::vector<RansacResult> res;
  };
  std::vector<std::unique_ptr<Chunk>> chunks;
  const bool essential = p.model == "essential";
  const auto verify = [&pairs, &p](Chunk& ch) {
    // 5-point E-RANSAC per pair on all CPU cores (independent, seeded per pair: the result does
    // not depend on the schedule or the chunking).
    const core::TraceRange r("match.ransac");
    ch.res.resize(ch.x1.size());
    core::parallel_for(ch.x1.size(), [&](std::size_t k) {
      ch.res[k] =
          verify_geometry(ch.x1[k], ch.x2[k], pairs[ch.kept[k]].first, pairs[ch.kept[k]].second, p);
    });
  };
  std::future<void> pending;  // the previous chunk's E-RANSAC
  // The top-2 of every query keypoint (16 bytes, both directions with mutual) is only needed until
  // the ratio / mutual test: run it in chunks of pairs and keep the putative matches. P9a: all
  // 43 k pairs of the 4-camera delivery_area at once held ~7 GB of top-2 on the host (and the
  // device). Pairs are independent, so the result does not depend on the chunking.
  std::size_t begin = 0;
  while (begin < pairs.size()) {
    std::size_t end = begin;
    std::size_t queries = 0;
    while (end < pairs.size() &&
           (end == begin || queries + features[static_cast<std::size_t>(pairs[end].first)].size() <=
                                max_queries_per_chunk)) {
      queries += features[static_cast<std::size_t>(pairs[end].first)].size();
      ++end;
    }
    const std::vector<std::pair<int, int>> chunk(pairs.begin() + static_cast<std::ptrdiff_t>(begin),
                                                 pairs.begin() + static_cast<std::ptrdiff_t>(end));
    const core::TraceRange chunk_range("match.chunk");
    std::vector<std::vector<Knn2>> fwd;
    {
      const core::TraceRange r("match.knn2");
      fwd = knn2_cuda(features, chunk);
    }
    std::vector<std::vector<Knn2>> back;
    if (p.mutual) {
      const core::TraceRange r("match.knn2_back");
      std::vector<std::pair<int, int>> rev;
      rev.reserve(chunk.size());
      for (const auto& [i, j] : chunk) rev.emplace_back(j, i);
      back = knn2_cuda(features, rev);
    }
    auto ch = std::make_unique<Chunk>();
    for (std::size_t c = 0; c < chunk.size(); ++c) {
      const std::size_t k = begin + c;
      std::vector<Match> m;
      for (std::size_t q = 0; q < fwd[c].size(); ++q) {
        const Knn2& r = fwd[c][q];
        if (r.best < 0) continue;
        if (!passes_ratio(r, p.ratio, words)) continue;
        if (p.mutual && back[c][static_cast<std::size_t>(r.best)].best != static_cast<int>(q)) {
          continue;
        }
        m.push_back({static_cast<int>(q), r.best, r.best_distance});
      }
      if (m.size() < static_cast<std::size_t>(p.min_inliers) || m.size() < 8) continue;
      const auto& fi = features[static_cast<std::size_t>(pairs[k].first)];
      const auto& fj = features[static_cast<std::size_t>(pairs[k].second)];
      Points2 a;
      Points2 b;
      a.reserve(m.size());
      b.reserve(m.size());
      for (const Match& mm : m) {
        const auto q = static_cast<std::size_t>(mm.query);
        const auto t = static_cast<std::size_t>(mm.train);
        a.emplace_back(fi.xy[2 * q], fi.xy[2 * q + 1]);
        b.emplace_back(fj.xy[2 * t], fj.xy[2 * t + 1]);
      }
      ch->kept.push_back(k);
      ch->putative.push_back(std::move(m));
      ch->x1.push_back(std::move(a));
      ch->x2.push_back(std::move(b));
      ch->seeds.push_back(pair_seed(p.seed, pairs[k].first, pairs[k].second));
    }
    if (essential) {
      if (pending.valid()) pending.get();  // one chunk verifying at a time; rethrows its error
      pending = std::async(std::launch::async, verify, std::ref(*ch));
    }
    chunks.push_back(std::move(ch));
    begin = end;
  }
  if (pending.valid()) pending.get();

  if (!essential) {
    // GPU F-RANSAC over all problems at once.
    std::vector<Points2> x1;
    std::vector<Points2> x2;
    std::vector<std::uint64_t> seeds;
    for (auto& ch : chunks) {
      for (std::size_t k = 0; k < ch->x1.size(); ++k) {
        x1.push_back(std::move(ch->x1[k]));
        x2.push_back(std::move(ch->x2[k]));
        seeds.push_back(ch->seeds[k]);
      }
    }
    RansacParams rp;
    rp.threshold_px = p.threshold_px;
    rp.confidence = p.confidence;
    rp.max_iters = p.max_iters;
    std::vector<RansacResult> res = ransac_fundamental_cuda(x1, x2, seeds, rp);
    std::size_t at = 0;
    for (auto& ch : chunks) {
      ch->res.assign(
          std::make_move_iterator(res.begin() + static_cast<std::ptrdiff_t>(at)),
          std::make_move_iterator(res.begin() + static_cast<std::ptrdiff_t>(at + ch->kept.size())));
      at += ch->kept.size();
    }
  }

  std::vector<PairMatches> out;
  for (const auto& ch : chunks) {
    for (std::size_t r = 0; r < ch->res.size(); ++r) {
      const RansacResult& rr = ch->res[r];
      if (rr.inliers < p.min_inliers) continue;
      PairMatches pm;
      pm.i = pairs[ch->kept[r]].first;
      pm.j = pairs[ch->kept[r]].second;
      pm.F = rr.F;
      pm.idx.reserve(2 * static_cast<std::size_t>(rr.inliers));
      for (std::size_t m = 0; m < ch->putative[r].size(); ++m) {
        if (rr.inlier[m] != 0) {
          pm.idx.push_back(ch->putative[r][m].query);
          pm.idx.push_back(ch->putative[r][m].train);
        }
      }
      out.push_back(std::move(pm));
    }
  }
  return out;
}

}  // namespace sfm::matching
