#pragma once
// Which files make up a dataset: the image, mask and calibration of every image in the configured
// range, for both layouts (config.hpp DatasetConfig::layout), and each image's camera.
//   numbered: <images_dir>/NNNN.ppm, <masks_dir>/NNNN.pgm, <calib_dir>/NNNN.txt, ids first..last.
//   folder:   every .jpg/.jpeg/.png/.ppm/.pgm in images_dir (any case), sorted by file name;
//             first..last index that list (last = -1: to the end). The mask of photo.jpg is
//             <masks_dir>/photo.png or photo.pgm, the calibration <calib_dir>/photo.txt.
// With dataset.camera set, no calibration files are read: every image gets P = K [I | 0].

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/data/krt.hpp"

namespace sfm::data {

// id: the file id (numbered) or the index in the sorted list (folder). name: the file name
// ("0007.ppm", "IMG_2041.JPG"), used by the exports. mask: empty when use_masks is false. calib:
// empty when dataset.camera is set.
struct ImageEntry {
  int id = 0;
  std::string name;
  std::filesystem::path image;
  std::filesystem::path mask;
  std::filesystem::path calib;
};

// Resolves the files of the configured range. Throws IoError when the folder has no images, the
// range runs past the last image, or (folder layout) a mask is missing.
std::vector<ImageEntry> list_images(const config::DatasetConfig& d);

// The camera of one image: its calibration file, or dataset.camera (EXIF read from the image
// file when camera.exif is set). width/height are the image's size (for the default centre and
// the EXIF focal length). Throws IoError on a missing or malformed calibration file.
Krt load_camera(const config::DatasetConfig& d, const ImageEntry& e, int width, int height);

// EXIF focal length of a JPEG, from its 35 mm-equivalent tag (FocalLengthIn35mmFilm) or from
// FocalLength in mm with sensor_width_mm > 0, converted to pixels for an image whose longer side
// is max_side. nullopt when the file is not a JPEG or has neither tag.
std::optional<double> exif_focal_px(const std::filesystem::path& jpeg, int max_side,
                                    double sensor_width_mm);

// EXIF orientation (1..8) of a JPEG, 1 when absent. Images are used as stored, so anything other
// than 1 means the photo is displayed rotated by viewers but read unrotated here.
int exif_orientation(const std::filesystem::path& jpeg);

}  // namespace sfm::data
