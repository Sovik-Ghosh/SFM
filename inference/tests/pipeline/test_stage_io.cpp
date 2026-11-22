#include <gtest/gtest.h>

#include <filesystem>

#include "sfm/core/error.hpp"
#include "sfm/pipeline/stage_io.hpp"
#include "sfm/recon/incremental.hpp"

#include "scene.hpp"

namespace fs = std::filesystem;
using namespace sfm::pipeline;  // NOLINT(google-build-using-namespace)

namespace {

class StageIo : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           (std::string("sfm_stage_io_") +
            ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  fs::path dir_;
};

sfm::recon::test::PipelineInput input() {
  sfm::eval::SyntheticConfig c;
  c.cameras = 4;
  c.points = 50;
  c.box_half = 20.0;
  auto in = sfm::recon::test::make_input(c);
  for (auto& f : in.features) {  // fill the remaining fields with distinct values
    for (std::size_t k = 0; k < f.size(); ++k) {
      f.response.push_back(0.5F * static_cast<float>(k));
      f.octave.push_back(static_cast<std::uint8_t>(k % 8));
      for (int w = 0; w < 8; ++w) {
        f.desc.push_back(static_cast<std::uint32_t>(k * 8 + static_cast<std::size_t>(w)));
      }
    }
  }
  return in;
}

}  // namespace

TEST_F(StageIo, EveryStageRoundTrips) {
  const auto in = input();
  save_features(dir_ / "features", in.features);
  const auto f = load_features(dir_ / "features", in.features.size());
  ASSERT_EQ(f.size(), in.features.size());
  for (std::size_t i = 0; i < f.size(); ++i) {
    EXPECT_EQ(f[i].xy, in.features[i].xy);
    EXPECT_EQ(f[i].angle, in.features[i].angle);
    EXPECT_EQ(f[i].response, in.features[i].response);
    EXPECT_EQ(f[i].octave, in.features[i].octave);
    EXPECT_EQ(f[i].desc, in.features[i].desc);
  }

  save_matches(dir_ / "matching", in.pairs);
  const auto m = load_matches(dir_ / "matching");
  ASSERT_EQ(m.size(), in.pairs.size());
  for (std::size_t k = 0; k < m.size(); ++k) {
    EXPECT_EQ(m[k].i, in.pairs[k].i);
    EXPECT_EQ(m[k].j, in.pairs[k].j);
    EXPECT_EQ(m[k].idx, in.pairs[k].idx);
    EXPECT_EQ(m[k].F, in.pairs[k].F);
  }

  save_tracks(dir_ / "tracks", in.tracks);
  const auto t = load_tracks(dir_ / "tracks");
  EXPECT_EQ(t.obs, in.tracks.obs);
  EXPECT_EQ(t.count, in.tracks.count);

  const sfm::recon::TrackIndex idx(in.tracks, in.features);
  const auto rec = sfm::recon::reconstruct_incremental(
      in.pairs, in.features, idx, in.K, sfm::recon::IncrementalParams{}, {}, nullptr);
  save_reconstruction(dir_ / "reconstruction", rec);
  const auto r = load_reconstruction(dir_ / "reconstruction", in.K);
  ASSERT_EQ(r.cameras.size(), rec.cameras.size());
  for (std::size_t c = 0; c < r.cameras.size(); ++c) {
    EXPECT_EQ(r.cameras[c].pose.R, rec.cameras[c].pose.R);
    EXPECT_EQ(r.cameras[c].pose.t, rec.cameras[c].pose.t);
    EXPECT_EQ(r.cameras[c].registered, rec.cameras[c].registered);
    EXPECT_EQ(r.cameras[c].K, in.K[c]);
  }
  EXPECT_EQ(r.points, rec.points);
  EXPECT_EQ(r.has_point, rec.has_point);
  EXPECT_EQ(r.order, rec.order);
}

TEST_F(StageIo, SiftDescriptorWidthRoundTrips) {
  sfm::features::Features f;
  f.xy = {1.0F, 2.0F, 3.0F, 4.0F};
  f.angle = {0.0F, 90.0F};
  f.response = {0.5F, 0.25F};
  f.octave = {0, 1};
  f.words = sfm::features::kSiftWords;
  for (std::uint32_t k = 0; k < 64; ++k) f.desc.push_back(k * 0x01010101U);
  save_features(dir_ / "features", {f});
  const auto g = load_features(dir_ / "features", 1);
  EXPECT_EQ(g[0].words, sfm::features::kSiftWords);
  EXPECT_EQ(g[0].desc, f.desc);
  EXPECT_EQ(g[0].descriptor(1)[0], 32U * 0x01010101U);
}

TEST_F(StageIo, MissingOrInconsistentFilesThrow) {
  EXPECT_THROW(load_features(dir_ / "features", 1), sfm::IoError);
  EXPECT_THROW(load_matches(dir_ / "matching"), sfm::IoError);
  const auto in = input();
  save_reconstruction(dir_ / "reconstruction", sfm::recon::Reconstruction{});
  EXPECT_THROW(load_reconstruction(dir_ / "reconstruction", in.K), sfm::IoError);  // 0 vs 4 cameras
}
