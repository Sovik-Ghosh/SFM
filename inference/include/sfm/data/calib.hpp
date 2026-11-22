#pragma once
// Bonn multi-view calibration files: a line "CONTOUR", then the 3x4 projection matrix P (row-major,
// whitespace separated) with P * X = d * u for a world point X and its pixel u.

#include <filesystem>

#include <Eigen/Core>

namespace sfm::data {

using Mat34 = Eigen::Matrix<double, 3, 4>;

// Throws IoError if the file is missing, the header is not CONTOUR, or there are not exactly
// 12 finite numbers after it.
Mat34 read_projection(const std::filesystem::path& path);

}  // namespace sfm::data
