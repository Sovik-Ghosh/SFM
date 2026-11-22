#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "sfm/core/arena.hpp"
#include "sfm/core/cuda_check.h"
#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/core/stream.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/dataset.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/krt.hpp"

#include "temp_dir.hpp"

namespace fs = std::filesystem;
using sfm::config::DatasetConfig;
using sfm::core::Logger;
using sfm::core::LogLevel;
using sfm::data::load_dataset;
using DatasetFiles = sfm::data::test::TempDirTest;

namespace {

DatasetConfig bunny(int first, int last, bool masks) {
  DatasetConfig c;
  c.root = fs::path(SFM_SOURCE_DIR) / "data/bunny";
  c.images_dir = "images";
  c.masks_dir = "silhouettes";
  c.calib_dir = "calib";
  c.first = first;
  c.last = last;
  c.use_masks = masks;
  return c;
}

}  // namespace

TEST(DatasetBunny, LoadsAll36ViewsWithMasksAndCalib) {
  std::ostringstream log_out;
  Logger log(LogLevel::kInfo, &log_out, false);
  const auto ds = load_dataset(bunny(0, 35, true), log);

  ASSERT_EQ(ds.count(), 36u);
  EXPECT_EQ(ds.ids.front(), 0);
  EXPECT_EQ(ds.ids.back(), 35);
  EXPECT_EQ(ds.width, 1024);
  EXPECT_EQ(ds.height, 768);
  EXPECT_EQ(ds.gray.size(), 36u * 1024 * 768);
  ASSERT_TRUE(ds.has_masks());
  EXPECT_EQ(ds.P.size(), 36u);
  ASSERT_EQ(ds.cameras.size(), 36u);
  for (std::size_t i = 0; i < ds.count(); ++i) {
    SCOPED_TRACE(testing::Message() << "camera " << ds.ids[i]);
    // K [R | t] is P up to a non-zero scale: compare after matching norm and sign.
    const sfm::data::Mat34 Q = sfm::data::compose_projection(ds.cameras[i]);
    const sfm::data::Mat34& P = ds.P[i];
    const double s = (P.cwiseProduct(Q).sum() < 0 ? -1.0 : 1.0) * P.norm() / Q.norm();
    EXPECT_LT((P - s * Q).norm() / P.norm(), 1e-12);
    // Bunny cameras sit on a hemisphere 57.0-64.8 units from the origin (numpy, design doc P1).
    EXPECT_GT(ds.cameras[i].centre().norm(), 56.0);
    EXPECT_LT(ds.cameras[i].centre().norm(), 66.0);
  }

  // Mask 0000: 48,661 object pixels (numpy: (silhouette == 0).sum()), stored as 255.
  std::size_t object = 0;
  for (std::size_t i = 0; i < ds.pixels(); ++i) object += ds.mask_image(0)[i] == 255;
  EXPECT_EQ(object, 48661u);

  // Last image lands in the last slot: sum of cv2 gray of 0035.ppm.
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < ds.pixels(); ++i) sum += ds.gray_image(35)[i];
  EXPECT_EQ(sum, 32604829u);

  EXPECT_NE(log_out.str().find("event=dataset.loaded"), std::string::npos);
  EXPECT_NE(log_out.str().find("images=36"), std::string::npos);
}

TEST(DatasetBunny, SubRangeWithoutMasks) {
  Logger log(LogLevel::kError, nullptr, false);
  const auto ds = load_dataset(bunny(3, 5, false), log);
  EXPECT_EQ(ds.ids, (std::vector<int>{3, 4, 5}));
  EXPECT_FALSE(ds.has_masks());
  const auto root = fs::path(SFM_SOURCE_DIR) / "data/bunny";
  EXPECT_EQ(ds.P[0], sfm::data::read_projection(root / "calib/0003.txt"));
  const auto img4 = sfm::data::load_gray(root / "images/0004.ppm");
  EXPECT_TRUE(std::equal(img4.pixels.begin(), img4.pixels.end(), ds.gray_image(1)));
}

TEST(DatasetBunny, UploadCopiesToArena) {
  Logger log(LogLevel::kError, nullptr, false);
  const auto ds = load_dataset(bunny(0, 1, true), log);
  sfm::core::Arena arena(8u << 20);
  sfm::core::Stream stream;
  const auto d = sfm::data::upload(ds, arena, stream);
  stream.synchronize();
  ASSERT_NE(d.masks, nullptr);
  EXPECT_EQ(arena.used() >= ds.gray.bytes() + ds.masks.bytes(), true);

  std::vector<std::uint8_t> back(ds.gray.size());
  CHECK(cudaMemcpy(back.data(), d.gray, back.size(), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::equal(back.begin(), back.end(), ds.gray.begin()));
  CHECK(cudaMemcpy(back.data(), d.masks, back.size(), cudaMemcpyDeviceToHost));
  EXPECT_TRUE(std::equal(back.begin(), back.end(), ds.masks.begin()));

  sfm::core::Arena small(1u << 20);
  EXPECT_THROW(sfm::data::upload(ds, small, stream), sfm::ArenaError);
}

TEST_F(DatasetFiles, MissingFilesAndSizeMismatchThrow) {
  const std::string calib = "CONTOUR\n1 0 0 0\n0 1 0 0\n0 0 1 0\n";
  write_pnm("images/0000.ppm", "P6", 2, 2, std::string(12, '\x10'));
  write_pnm("images/0001.ppm", "P6", 2, 1, std::string(6, '\x10'));  // wrong size
  write("calib/0000.txt", calib);
  write("calib/0001.txt", calib);

  DatasetConfig c;
  c.root = dir_;
  c.images_dir = "images";
  c.masks_dir = "silhouettes";
  c.calib_dir = "calib";
  c.use_masks = false;
  Logger log(LogLevel::kError, nullptr, false);

  c.first = c.last = 0;
  EXPECT_EQ(load_dataset(c, log).count(), 1u);
  c.last = 1;
  EXPECT_THROW(load_dataset(c, log), sfm::IoError);  // 0001 is 2x1
  c.last = 0;
  c.use_masks = true;
  EXPECT_THROW(load_dataset(c, log), sfm::IoError);  // no silhouettes/0000.pgm
  c.use_masks = false;
  fs::remove(dir_ / "calib/0000.txt");
  EXPECT_THROW(load_dataset(c, log), sfm::IoError);
  c.first = 5;
  c.last = 4;
  EXPECT_THROW(load_dataset(c, log), sfm::IoError);
}
