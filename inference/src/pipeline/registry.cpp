#include "sfm/pipeline/registry.hpp"

#include <memory>
#include <string>
#include <vector>

#include "sfm/ba/adjust.hpp"
#include "sfm/core/device_info.hpp"
#include "sfm/core/error.hpp"
#include "sfm/features/orb.hpp"
#include "sfm/features/sift.hpp"
#include "sfm/matching/verify.hpp"

#if SFM_WITH_CUDA
#include "sfm/ba/ba_cuda.hpp"
#include "sfm/features/orb_cuda.hpp"
#include "sfm/features/sift_cuda.hpp"
#include "sfm/matching/matching_cuda.hpp"
#endif

namespace sfm::pipeline {

namespace {

// The cached detector that fits (same image size and parameters), else a new one; keeps at most
// kMaxDetectors (device memory), dropping the oldest.
constexpr std::size_t kMaxDetectors = 4;

template <class Detector, class Fits, class Make>
Detector& cached_detector(std::vector<std::unique_ptr<Detector>>& cache, const Fits& fits,
                          const Make& make) {
  for (auto& d : cache) {
    if (fits(*d)) return *d;
  }
  if (cache.size() == kMaxDetectors) cache.erase(cache.begin());
  cache.push_back(make());
  return *cache.back();
}

features::SiftParams sift_params(const config::FeaturesConfig& c) {
  features::SiftParams p;
  p.max_features = c.max_features;
  p.octaves = c.sift.octaves;
  p.octave_resolution = c.sift.octave_resolution;
  p.first_octave = c.sift.first_octave;
  p.peak_threshold = static_cast<float>(c.sift.peak_threshold);
  p.edge_threshold = static_cast<float>(c.sift.edge_threshold);
  p.max_orientations = c.sift.max_orientations;
  return p;
}

template <class Map>
const typename Map::mapped_type& find(const Map& m, const std::string& name, const char* path) {
  const auto it = m.find(name);
  if (it != m.end()) return it->second;
  std::string known;
  for (const auto& kv : m) known += (known.empty() ? "" : ", ") + kv.first;
  throw ConfigError(std::string(path) + ": unknown method '" + name + "' (known: " + known + ")");
}

Registry make_default() {
  Registry r;
  r.add_features("orb_cpu", [](const data::Image8& gray, const data::Image8* mask,
                               const config::FeaturesConfig& c) {
    features::OrbParams p;
    p.max_features = c.max_features;
    p.levels = c.pyramid_levels;
    p.scale = c.scale;
    p.fast_threshold = c.fast_threshold;
    p.edge_border = c.edge_border;
    return features::detect_orb(gray, mask, p);
  });
#if SFM_WITH_CUDA
  // OrbCuda detectors kept across images (device buffers are reused), one per image size: photo
  // sets mix portrait and landscape. At most kMaxDetectors sizes; the oldest is dropped.
  auto cached = std::make_shared<std::vector<std::unique_ptr<features::OrbCuda>>>();
  r.add_features("orb_cuda", [cached](const data::Image8& gray, const data::Image8* mask,
                                      const config::FeaturesConfig& c) {
    features::OrbParams p;
    p.max_features = c.max_features;
    p.levels = c.pyramid_levels;
    p.scale = c.scale;
    p.fast_threshold = c.fast_threshold;
    p.edge_border = c.edge_border;
    const auto fits = [&](const features::OrbCuda& d) {
      const auto& q = d.params();
      return d.width() == gray.width && d.height() == gray.height &&
             q.max_features == p.max_features && q.levels == p.levels && q.scale == p.scale &&
             q.fast_threshold == p.fast_threshold && q.edge_border == p.edge_border;
    };
    const auto make = [&] {
      return std::make_unique<features::OrbCuda>(p, gray.width, gray.height);
    };
    return cached_detector(*cached, fits, make).detect(gray, mask);
  });
#endif
  r.add_features("sift_cpu", [](const data::Image8& gray, const data::Image8* mask,
                                const config::FeaturesConfig& c) {
    return features::detect_sift(gray, mask, sift_params(c));
  });
#if SFM_WITH_CUDA
  // SiftCuda detectors, one per image size, as for ORB.
  auto sift_cached = std::make_shared<std::vector<std::unique_ptr<features::SiftCuda>>>();
  r.add_features("sift_cuda", [sift_cached](const data::Image8& gray, const data::Image8* mask,
                                            const config::FeaturesConfig& c) {
    const features::SiftParams p = sift_params(c);
    const auto fits = [&](const features::SiftCuda& d) {
      const auto& q = d.params();
      return d.width() == gray.width && d.height() == gray.height &&
             q.max_features == p.max_features && q.octaves == p.octaves &&
             q.octave_resolution == p.octave_resolution && q.first_octave == p.first_octave &&
             q.peak_threshold == p.peak_threshold && q.edge_threshold == p.edge_threshold &&
             q.max_orientations == p.max_orientations;
    };
    const auto make = [&] {
      return std::make_unique<features::SiftCuda>(p, gray.width, gray.height);
    };
    return cached_detector(*sift_cached, fits, make).detect(gray, mask);
  });
#endif
  r.add_matching("bf_cpu", [](const std::vector<features::Features>& f,
                              const config::MatchingConfig& c,
                              const std::vector<Eigen::Matrix3d>& K, std::uint64_t seed) {
    matching::VerifyParams p;
    p.ratio = c.ratio;
    p.mutual = c.mutual;
    p.threshold_px = c.ransac.threshold_px;
    p.confidence = c.ransac.confidence;
    p.max_iters = static_cast<int>(c.ransac.max_iters);
    p.min_inliers = static_cast<int>(c.min_inliers);
    p.seed = seed;
    p.model = c.ransac.model;
    p.K = &K;
    return matching::verify_pairs(f,
                                  matching::select_pairs(static_cast<int>(f.size()), c.pairs.mode,
                                                         static_cast<int>(c.pairs.window),
                                                         static_cast<int>(c.pairs.keyframe_stride)),
                                  p);
  });
#if SFM_WITH_CUDA
  r.add_matching("bf_cuda",
                 [](const std::vector<features::Features>& f, const config::MatchingConfig& c,
                    const std::vector<Eigen::Matrix3d>& K, std::uint64_t seed) {
                   matching::VerifyParams p;
                   p.ratio = c.ratio;
                   p.mutual = c.mutual;
                   p.threshold_px = c.ransac.threshold_px;
                   p.confidence = c.ransac.confidence;
                   p.max_iters = static_cast<int>(c.ransac.max_iters);
                   p.min_inliers = static_cast<int>(c.min_inliers);
                   p.seed = seed;
                   p.model = c.ransac.model;
                   p.K = &K;
                   return matching::verify_pairs_cuda(
                       f,
                       matching::select_pairs(static_cast<int>(f.size()), c.pairs.mode,
                                              static_cast<int>(c.pairs.window),
                                              static_cast<int>(c.pairs.keyframe_stride)),
                       p);
                 });
#endif
  r.add_ba("lm_schur_cpu", [](recon::Reconstruction& rec, const recon::TrackIndex& index,
                              const config::BaConfig& c, const std::vector<int>& free_images) {
    ba::Options o;
    o.huber_px = c.huber_px;
    o.max_iters = static_cast<int>(c.max_iters);
    return ba::bundle_adjust(rec, index, o, &ba::solve, free_images);
  });
#if SFM_WITH_CUDA
  r.add_ba("lm_schur_cuda", [](recon::Reconstruction& rec, const recon::TrackIndex& index,
                               const config::BaConfig& c, const std::vector<int>& free_images) {
    ba::Options o;
    o.huber_px = c.huber_px;
    o.max_iters = static_cast<int>(c.max_iters);
    return ba::bundle_adjust(rec, index, o, &ba::solve_cuda, free_images);
  });
#endif
  return r;
}

}  // namespace

const FeatureMethod& Registry::features(const std::string& name) const {
  return find(features_, name, "features.method");
}
const MatchMethod& Registry::matching(const std::string& name) const {
  return find(matching_, name, "matching.method");
}
const BaMethod& Registry::ba(const std::string& name) const { return find(ba_, name, "ba.method"); }

Backend detect_backend() {
  Backend b;
  if (core::device_count() > 0) {
    const core::DeviceInfo info = core::query_device(0);
    b.device = info.name;
    std::string why;
    if (core::kernels_runnable(0, why)) {
      b.gpu = true;
    } else {
      // e.g. a build for sm_120 only on an RTX 3060 (sm_86): run on the CPU instead of failing.
      b.unusable = info.name + " (sm_" + std::to_string(info.cc_major) +
                   std::to_string(info.cc_minor) + "): " + why +
                   "; rebuild with -DCMAKE_CUDA_ARCHITECTURES=" + std::to_string(info.cc_major) +
                   std::to_string(info.cc_minor);
    }
  }
  return b;
}

config::Config resolve_auto_methods(const config::Config& cfg, bool gpu) {
  config::Config c = cfg;
  // SIFT since P7: COLMAP's accuracy on bunny / ETH3D, where ORB loses cameras (README, P7).
  if (c.features.method == "auto") c.features.method = gpu ? "sift_cuda" : "sift_cpu";
  if (c.matching.method == "auto") c.matching.method = gpu ? "bf_cuda" : "bf_cpu";
  if (c.ba.method == "auto") c.ba.method = gpu ? "lm_schur_cuda" : "lm_schur_cpu";
  return c;
}

namespace {

std::string swap_suffix(const std::string& m, const std::string& from, const std::string& to) {
  if (m.size() > from.size() && m.compare(m.size() - from.size(), from.size(), from) == 0) {
    return m.substr(0, m.size() - from.size()) + to;
  }
  return m;
}

}  // namespace

config::Config use_backend(const config::Config& cfg, bool gpu) {
  config::Config c = resolve_auto_methods(cfg, gpu);
  const std::string from = gpu ? "_cpu" : "_cuda";
  const std::string to = gpu ? "_cuda" : "_cpu";
  c.features.method = swap_suffix(c.features.method, from, to);
  c.matching.method = swap_suffix(c.matching.method, from, to);
  c.ba.method = swap_suffix(c.ba.method, from, to);
  return c;
}

bool built_with_cuda() { return SFM_WITH_CUDA != 0; }

const Registry& default_registry() {
  static const Registry kRegistry = make_default();
  return kRegistry;
}

}  // namespace sfm::pipeline
