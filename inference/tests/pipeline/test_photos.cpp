// pipeline::photo_config: the defaults of `sfm_run --images` and epipole.reconstruct().
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "sfm/core/error.hpp"
#include "sfm/pipeline/photos.hpp"

namespace fs = std::filesystem;

TEST(Photos, ConfigForAFolderOfPhotos) {
  const fs::path dir = fs::temp_directory_path() / "sfm_photos_test" / "my holiday";
  fs::remove_all(dir.parent_path());
  fs::create_directories(dir);
  fs::create_directories(dir.parent_path() / "masks");
  sfm::pipeline::PhotoOptions o;
  o.images = dir;
  o.output_dir = dir.parent_path() / "out";
  auto c = sfm::pipeline::photo_config(o);
  EXPECT_EQ(c.run.name, "my_holiday");  // the folder's name, made safe
  EXPECT_EQ(c.dataset.root, dir);
  EXPECT_EQ(c.dataset.layout, "folder");
  EXPECT_EQ(c.dataset.last, -1);
  EXPECT_FALSE(c.dataset.use_masks);
  EXPECT_TRUE(c.dataset.camera.exif);
  EXPECT_TRUE(c.ba.refine_focal);
  EXPECT_TRUE(c.output.colmap_scene);
  EXPECT_EQ(c.features.method, "auto");
  EXPECT_EQ(c.matching.method, "auto");
  EXPECT_EQ(c.ba.method, "auto");
  EXPECT_EQ(c.matching.pairs.mode, "exhaustive");
  EXPECT_EQ(c.features.sift.peak_threshold, 0.0015);  // the tuned SIFT config

  o.fx = 3000.0;
  o.masks = dir.parent_path() / "masks";
  o.mask_object_white = false;
  o.sequential = true;
  o.name = "run1";
  c = sfm::pipeline::photo_config(o);
  EXPECT_EQ(c.run.name, "run1");
  EXPECT_FALSE(c.dataset.camera.exif);
  EXPECT_DOUBLE_EQ(c.dataset.camera.fx, 3000.0);
  EXPECT_TRUE(c.dataset.use_masks);
  EXPECT_FALSE(c.dataset.mask_object_white);
  EXPECT_EQ(c.matching.pairs.mode, "sequential");

  o.images = dir.parent_path() / "missing";
  EXPECT_THROW(sfm::pipeline::photo_config(o), sfm::IoError);
  fs::remove_all(dir.parent_path());
}
