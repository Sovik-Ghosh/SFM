#include "sfm/pipeline/stage_io.hpp"

#include <string>

#include "sfm/core/error.hpp"
#include "sfm/core/npy.hpp"

namespace sfm::pipeline {

namespace fs = std::filesystem;
using core::npy_read;
using core::npy_write;
using core::NpyArray;

namespace {

std::string image_name(std::size_t i) {
  std::string s = std::to_string(i);
  return std::string(s.size() < 4 ? 4 - s.size() : 0, '0') + s;
}

template <class T>
std::vector<T> read(const fs::path& file, std::size_t expect_dims,
                    std::vector<std::size_t>* shape = nullptr) {
  const NpyArray a = npy_read(file);
  if (a.shape.size() != expect_dims) throw IoError(file.string() + ": unexpected rank");
  if (shape != nullptr) *shape = a.shape;
  return a.to_vector<T>();
}

}  // namespace

void save_features(const fs::path& dir, const std::vector<features::Features>& f) {
  fs::create_directories(dir);
  for (std::size_t i = 0; i < f.size(); ++i) {
    const auto n = f[i].size();
    const fs::path b = dir / image_name(i);
    npy_write(b.string() + "_xy.npy", NpyArray::from<float>({n, 2}, f[i].xy));
    npy_write(b.string() + "_angle.npy", NpyArray::from<float>({n}, f[i].angle));
    npy_write(b.string() + "_response.npy", NpyArray::from<float>({n}, f[i].response));
    npy_write(b.string() + "_octave.npy", NpyArray::from<std::uint8_t>({n}, f[i].octave));
    npy_write(b.string() + "_desc.npy",
              NpyArray::from<std::uint32_t>({n, static_cast<std::size_t>(f[i].words)}, f[i].desc));
  }
}

std::vector<features::Features> load_features(const fs::path& dir, std::size_t images) {
  std::vector<features::Features> out(images);
  for (std::size_t i = 0; i < images; ++i) {
    const std::string b = (dir / image_name(i)).string();
    auto& f = out[i];
    f.xy = read<float>(b + "_xy.npy", 2);
    f.angle = read<float>(b + "_angle.npy", 1);
    f.response = read<float>(b + "_response.npy", 1);
    f.octave = read<std::uint8_t>(b + "_octave.npy", 1);
    f.desc = read<std::uint32_t>(b + "_desc.npy", 2);
    const std::size_t n = f.angle.size();
    // Descriptor width from the file: 8 words (ORB) or 32 (SIFT); an empty image keeps ORB's.
    if (n > 0) f.words = static_cast<int>(f.desc.size() / n);
    if ((f.words != features::kDescriptorWords && f.words != features::kSiftWords) ||
        f.xy.size() != 2 * n || f.response.size() != n || f.octave.size() != n ||
        f.desc.size() != static_cast<std::size_t>(f.words) * n) {
      throw IoError(b + "_*.npy: inconsistent keypoint counts");
    }
  }
  return out;
}

void save_matches(const fs::path& dir, const std::vector<matching::PairMatches>& p) {
  fs::create_directories(dir);
  std::vector<std::int32_t> pairs;
  std::vector<std::int32_t> idx;
  std::vector<double> F;
  for (const auto& m : p) {
    pairs.insert(pairs.end(), {m.i, m.j, static_cast<std::int32_t>(m.size())});
    idx.insert(idx.end(), m.idx.begin(), m.idx.end());
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) F.push_back(m.F(r, c));
    }
  }
  npy_write(dir / "pairs.npy", NpyArray::from<std::int32_t>({p.size(), 3}, pairs));
  npy_write(dir / "idx.npy", NpyArray::from<std::int32_t>({idx.size() / 2, 2}, idx));
  npy_write(dir / "F.npy", NpyArray::from<double>({p.size(), 3, 3}, F));
}

std::vector<matching::PairMatches> load_matches(const fs::path& dir) {
  const auto pairs = read<std::int32_t>(dir / "pairs.npy", 2);
  const auto idx = read<std::int32_t>(dir / "idx.npy", 2);
  const auto F = read<double>(dir / "F.npy", 3);
  const std::size_t n = pairs.size() / 3;
  if (F.size() != 9 * n) throw IoError((dir / "F.npy").string() + ": pair count mismatch");
  std::vector<matching::PairMatches> out(n);
  std::size_t at = 0;
  for (std::size_t k = 0; k < n; ++k) {
    auto& m = out[k];
    m.i = pairs[3 * k];
    m.j = pairs[3 * k + 1];
    const auto count = static_cast<std::size_t>(pairs[3 * k + 2]);
    if (2 * (at + count) > idx.size()) throw IoError((dir / "idx.npy").string() + ": too short");
    m.idx.assign(idx.begin() + static_cast<std::ptrdiff_t>(2 * at),
                 idx.begin() + static_cast<std::ptrdiff_t>(2 * (at + count)));
    at += count;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m.F(r, c) = F[9 * k + static_cast<std::size_t>(3 * r + c)];
    }
  }
  if (2 * at != idx.size()) throw IoError((dir / "idx.npy").string() + ": too long");
  return out;
}

void save_tracks(const fs::path& dir, const matching::Tracks& t) {
  fs::create_directories(dir);
  npy_write(dir / "obs.npy", NpyArray::from<std::int32_t>({t.size(), 3}, t.obs));
}

matching::Tracks load_tracks(const fs::path& dir) {
  matching::Tracks t;
  t.obs = read<std::int32_t>(dir / "obs.npy", 2);
  t.count = t.obs.empty() ? 0 : t.obs[t.obs.size() - 3] + 1;
  return t;
}

void save_reconstruction(const fs::path& dir, const recon::Reconstruction& r) {
  fs::create_directories(dir);
  std::vector<double> R;
  std::vector<double> t;
  std::vector<std::uint8_t> reg;
  for (const auto& c : r.cameras) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) R.push_back(c.pose.R(i, j));
      t.push_back(c.pose.t(i));
    }
    reg.push_back(c.registered ? 1 : 0);
  }
  std::vector<double> X;
  for (const auto& p : r.points) X.insert(X.end(), {p.x(), p.y(), p.z()});
  const std::size_t n = r.cameras.size();
  npy_write(dir / "R.npy", NpyArray::from<double>({n, 3, 3}, R));
  npy_write(dir / "t.npy", NpyArray::from<double>({n, 3}, t));
  npy_write(dir / "registered.npy", NpyArray::from<std::uint8_t>({n}, reg));
  std::vector<std::int32_t> order(r.order.begin(), r.order.end());
  npy_write(dir / "order.npy", NpyArray::from<std::int32_t>({order.size()}, order));
  npy_write(dir / "points.npy", NpyArray::from<double>({r.points.size(), 3}, X));
  npy_write(dir / "has_point.npy", NpyArray::from<std::uint8_t>({r.has_point.size()}, r.has_point));
  npy_write(dir / "obs_outlier.npy",
            NpyArray::from<std::uint8_t>({r.obs_outlier.size()}, r.obs_outlier));
}

recon::Reconstruction load_reconstruction(const fs::path& dir,
                                          const std::vector<Eigen::Matrix3d>& K) {
  recon::Reconstruction r;
  const auto R = read<double>(dir / "R.npy", 3);
  const auto t = read<double>(dir / "t.npy", 2);
  const auto reg = read<std::uint8_t>(dir / "registered.npy", 1);
  if (reg.size() != K.size() || R.size() != 9 * K.size() || t.size() != 3 * K.size()) {
    throw IoError(dir.string() + ": camera count differs from the dataset");
  }
  r.cameras.resize(K.size());
  for (std::size_t c = 0; c < K.size(); ++c) {
    r.cameras[c].K = K[c];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.cameras[c].pose.R(i, j) = R[9 * c + static_cast<std::size_t>(3 * i + j)];
      }
      r.cameras[c].pose.t(i) = t[3 * c + static_cast<std::size_t>(i)];
    }
    r.cameras[c].registered = reg[c] != 0;
  }
  const auto order = read<std::int32_t>(dir / "order.npy", 1);
  r.order.assign(order.begin(), order.end());
  const auto X = read<double>(dir / "points.npy", 2);
  r.has_point = read<std::uint8_t>(dir / "has_point.npy", 1);
  if (X.size() != 3 * r.has_point.size()) throw IoError(dir.string() + ": point count mismatch");
  for (std::size_t k = 0; k < r.has_point.size(); ++k) {
    r.points.emplace_back(X[3 * k], X[3 * k + 1], X[3 * k + 2]);
  }
  if (fs::exists(dir / "obs_outlier.npy")) {  // runs before the outlier flags have none
    r.obs_outlier = read<std::uint8_t>(dir / "obs_outlier.npy", 1);
  }
  return r;
}

}  // namespace sfm::pipeline
