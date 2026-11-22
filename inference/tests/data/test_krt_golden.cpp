// decompose_projection on the 36 bunny cameras equals scipy's RQ result (tools/make_golden_p1.py).
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "sfm/core/npy.hpp"
#include "sfm/data/calib.hpp"
#include "sfm/data/krt.hpp"

TEST(KrtGolden, BunnyCamerasMatchScipy) {
  const std::filesystem::path src(SFM_SOURCE_DIR);
  const auto golden = src / "inference/tests/golden/p1";
  const auto K = sfm::core::npy_read(golden / "krt_K.npy").to_vector<double>();
  const auto R = sfm::core::npy_read(golden / "krt_R.npy").to_vector<double>();
  const auto t = sfm::core::npy_read(golden / "krt_t.npy").to_vector<double>();
  ASSERT_EQ(K.size(), 36u * 9);

  for (std::size_t c = 0; c < 36; ++c) {
    SCOPED_TRACE(c);
    std::string name = std::to_string(c);
    name.insert(0, 4 - name.size(), '0');
    const auto got = sfm::data::decompose_projection(
        sfm::data::read_projection(src / "data/bunny/calib" / (name + ".txt")));
    // npy is row-major; Eigen's Map default is column-major -> map as RowMajor.
    using RowMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    const Eigen::Matrix3d K_ref = Eigen::Map<const RowMat3>(&K[9 * c]);
    const Eigen::Matrix3d R_ref = Eigen::Map<const RowMat3>(&R[9 * c]);
    const Eigen::Vector3d t_ref = Eigen::Map<const Eigen::Vector3d>(&t[3 * c]);
    EXPECT_LT((got.K - K_ref).norm() / K_ref.norm(), 1e-9);
    EXPECT_LT((got.R - R_ref).norm(), 1e-9);
    EXPECT_LT((got.t - t_ref).norm() / t_ref.norm(), 1e-9);
  }
}
