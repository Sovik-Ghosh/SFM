// Dataset file resolution (both layouts), the shared camera, EXIF focal length, JPEG input and the
// mask polarity.
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/data/image_io.hpp"
#include "sfm/data/image_list.hpp"

#include "temp_dir.hpp"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace fs = std::filesystem;
using sfm::config::DatasetConfig;
using sfm::data::list_images;
using sfm::data::load_camera;
using ImageList = sfm::data::test::TempDirTest;

namespace {

void append(std::vector<std::uint8_t>* out, const void* data, int size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  out->insert(out->end(), p, p + size);
}

// A w x h gray JPEG (stb), optionally with an EXIF APP1 segment after SOI holding Orientation,
// FocalLengthIn35mmFilm (0: absent) and FocalLength in mm (0: absent). Little-endian TIFF.
std::string jpeg(int w, int h, int orientation, int f35, double focal_mm) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(w * h));
  for (std::size_t i = 0; i < px.size(); ++i) px[i] = static_cast<std::uint8_t>(i * 7);
  std::vector<std::uint8_t> j;
  const auto sink = [](void* ctx, void* d, int n) {
    append(static_cast<std::vector<std::uint8_t>*>(ctx), d, n);
  };
  stbi_write_jpg_to_func(sink, &j, w, h, 1, px.data(), 90);
  if (orientation == 0 && f35 == 0 && focal_mm == 0.0) return std::string(j.begin(), j.end());

  std::vector<std::uint8_t> t = {'I', 'I', 42, 0, 8, 0, 0, 0};
  const auto u16 = [&](std::uint32_t v) {
    t.push_back(v & 0xFF);
    t.push_back((v >> 8) & 0xFF);
  };
  const auto u32 = [&](std::uint32_t v) {
    u16(v & 0xFFFF);
    u16(v >> 16);
  };
  // IFD0 at 8: Orientation, ExifIFD pointer. 2 entries -> 2 + 24 + 4 = 30 bytes -> sub-IFD at 38.
  u16(2);
  u16(0x0112), u16(3), u32(1), u16(static_cast<std::uint32_t>(orientation)), u16(0);
  u16(0x8769), u16(4), u32(1), u32(38);
  u32(0);
  // Exif sub-IFD at 38: FocalLengthIn35mmFilm, FocalLength (rational at 38 + 30 = 68).
  u16(2);
  u16(0xA405), u16(3), u32(1), u16(static_cast<std::uint32_t>(f35)), u16(0);
  u16(0x920A), u16(5), u32(1), u32(68);
  u32(0);
  u32(static_cast<std::uint32_t>(focal_mm * 100)), u32(100);

  std::vector<std::uint8_t> app1 = {0xFF, 0xE1, 0, 0, 'E', 'x', 'i', 'f', 0, 0};
  app1.insert(app1.end(), t.begin(), t.end());
  const std::size_t len = app1.size() - 2;
  app1[2] = static_cast<std::uint8_t>(len >> 8);
  app1[3] = static_cast<std::uint8_t>(len & 0xFF);
  j.insert(j.begin() + 2, app1.begin(), app1.end());
  return std::string(j.begin(), j.end());
}

}  // namespace

TEST_F(ImageList, NumberedLayoutMatchesTheBunnyNames) {
  DatasetConfig d;
  d.root = dir_;
  d.images_dir = "images";
  d.masks_dir = "silhouettes";
  d.calib_dir = "calib";
  d.first = 7;
  d.last = 8;
  const auto e = list_images(d);
  ASSERT_EQ(e.size(), 2u);
  EXPECT_EQ(e[0].id, 7);
  EXPECT_EQ(e[0].name, "0007.ppm");
  EXPECT_EQ(e[0].image, dir_ / "images/0007.ppm");
  EXPECT_EQ(e[0].mask, dir_ / "silhouettes/0007.pgm");
  EXPECT_EQ(e[1].calib, dir_ / "calib/0008.txt");
}

TEST_F(ImageList, FolderLayoutSortsAnyNameAndExtension) {
  write("photos/IMG_10.JPG", jpeg(8, 6, 0, 0, 0));
  write("photos/IMG_02.jpeg", jpeg(8, 6, 0, 0, 0));
  write_pnm("photos/a.ppm", "P6", 8, 6, std::string(std::size_t{144}, 'x'));
  write("photos/notes.txt", "not an image");
  write("masks/IMG_02.png", "");
  write("masks/IMG_10.pgm", "");
  write("masks/a.PNG", "");
  DatasetConfig d;
  d.root = dir_;
  d.images_dir = "photos";
  d.masks_dir = "masks";
  d.calib_dir = "calib";
  d.layout = "folder";
  d.first = 0;
  d.last = -1;
  const auto e = list_images(d);
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e[0].name, "IMG_02.jpeg");
  EXPECT_EQ(e[1].name, "IMG_10.JPG");
  EXPECT_EQ(e[2].name, "a.ppm");
  EXPECT_EQ(e[0].mask, dir_ / "masks/IMG_02.png");
  EXPECT_EQ(e[1].mask, dir_ / "masks/IMG_10.pgm");
  EXPECT_EQ(e[2].mask, dir_ / "masks/a.PNG");
  EXPECT_EQ(e[1].calib, dir_ / "calib/IMG_10.txt");
  EXPECT_EQ(e[2].id, 2);

  d.first = 1;
  d.last = 1;
  ASSERT_EQ(list_images(d).size(), 1u);
  EXPECT_EQ(list_images(d)[0].name, "IMG_10.JPG");
  d.last = 3;  // past the end
  EXPECT_THROW(list_images(d), sfm::IoError);
  d.first = 0;
  d.last = -1;
  fs::remove(dir_ / "masks/a.PNG");
  EXPECT_THROW(list_images(d), sfm::IoError);  // missing mask
  d.use_masks = false;
  EXPECT_EQ(list_images(d).size(), 3u);
  EXPECT_TRUE(list_images(d)[0].mask.empty());
}

TEST_F(ImageList, EmptyFolderThrows) {
  fs::create_directories(dir_ / "photos");
  DatasetConfig d;
  d.root = dir_;
  d.images_dir = "photos";
  d.layout = "folder";
  d.last = -1;
  d.use_masks = false;
  EXPECT_THROW(list_images(d), sfm::IoError);
}

TEST_F(ImageList, SharedCameraNeedsNoCalibrationFiles) {
  DatasetConfig d;
  d.root = dir_;
  d.camera.fx = 900.0;
  const auto e = list_images(d);
  EXPECT_TRUE(e[0].calib.empty());
  const auto k = load_camera(d, e[0], 640, 480);
  EXPECT_DOUBLE_EQ(k.K(0, 0), 900.0);
  EXPECT_DOUBLE_EQ(k.K(1, 1), 900.0);  // fy defaults to fx
  EXPECT_DOUBLE_EQ(k.K(0, 2), 319.5);  // centre, pixel centres at integers
  EXPECT_DOUBLE_EQ(k.K(1, 2), 239.5);
  EXPECT_TRUE(k.R.isIdentity());
  d.camera.fy = 910.0;
  d.camera.cx = 300.0;
  d.camera.cy = 250.0;
  const auto k2 = load_camera(d, e[0], 640, 480);
  EXPECT_DOUBLE_EQ(k2.K(1, 1), 910.0);
  EXPECT_DOUBLE_EQ(k2.K(0, 2), 300.0);
  EXPECT_DOUBLE_EQ(k2.K(1, 2), 250.0);
}

TEST_F(ImageList, ExifFocalLength) {
  const auto with35 = write("a.jpg", jpeg(64, 48, 6, 27, 4.2));
  const auto mm_only = write("b.jpg", jpeg(64, 48, 1, 0, 4.2));
  const auto plain = write("c.jpg", jpeg(64, 48, 0, 0, 0));
  // 27 mm equivalent: the 64 px side spans 36 mm -> 27 / 36 * 64 = 48 px.
  EXPECT_NEAR(sfm::data::exif_focal_px(with35, 64, 0.0).value_or(-1.0), 48.0, 1e-9);
  // FocalLength alone needs the sensor width: 4.2 mm / 6.4 mm * 64 px = 42 px.
  EXPECT_FALSE(sfm::data::exif_focal_px(mm_only, 64, 0.0).has_value());
  EXPECT_NEAR(sfm::data::exif_focal_px(mm_only, 64, 6.4).value_or(-1.0), 42.0, 1e-9);
  EXPECT_FALSE(sfm::data::exif_focal_px(plain, 64, 6.4).has_value());
  EXPECT_EQ(sfm::data::exif_orientation(with35), 6);
  EXPECT_EQ(sfm::data::exif_orientation(plain), 1);

  DatasetConfig d;
  d.root = dir_;
  d.images_dir = ".";
  d.layout = "folder";
  d.last = -1;
  d.use_masks = false;
  d.camera.exif = true;
  const auto e = list_images(d);
  ASSERT_EQ(e.size(), 3u);
  EXPECT_NEAR(load_camera(d, e[0], 64, 48).K(0, 0), 48.0, 1e-9);
  EXPECT_NEAR(load_camera(d, e[2], 64, 48).K(0, 0), 1.2 * 64, 1e-9);  // no EXIF: COLMAP's default
}

TEST_F(ImageList, JpegLoadsAsGray) {
  const auto p = write("a.jpg", jpeg(16, 12, 6, 27, 0));  // with an EXIF segment
  const auto g = sfm::data::load_gray(p);
  EXPECT_EQ(g.width, 16);
  EXPECT_EQ(g.height, 12);
  EXPECT_EQ(g.size(), 16u * 12u);
}

TEST_F(ImageList, MaskPolarity) {
  const auto p = write_pnm("m.pgm", "P5", 3, 1, std::string("\x00\x80\xff", 3));
  const auto black = sfm::data::load_mask(p);
  EXPECT_EQ(black.pixels, (std::vector<std::uint8_t>{255, 0, 0}));
  const auto white = sfm::data::load_mask(p, true);
  EXPECT_EQ(white.pixels, (std::vector<std::uint8_t>{0, 255, 255}));
}
