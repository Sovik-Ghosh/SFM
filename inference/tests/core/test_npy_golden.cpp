// Interop with NumPy: files written by np.save (tools/make_golden_p1.py) are read correctly, and
// our writer produces the same bytes as np.save for the same array.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "sfm/core/npy.hpp"

namespace fs = std::filesystem;
using sfm::core::DType;
using sfm::core::npy_read;
using sfm::core::npy_write;
using sfm::core::NpyArray;

namespace {

const fs::path kGolden = fs::path(SFM_GOLDEN_DIR) / "p1";

std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Reads <name>.npy written by NumPy, checks it equals `expected`, then writes `expected` ourselves
// and checks the file is byte-identical to NumPy's.
template <class T>
void check_case(const std::string& name, const std::vector<std::size_t>& shape,
                const std::vector<T>& expected) {
  SCOPED_TRACE(name);
  const fs::path numpy_file = kGolden / "npy" / (name + ".npy");
  const NpyArray a = npy_read(numpy_file);
  EXPECT_EQ(a.dtype, sfm::core::dtype_of<T>());
  EXPECT_EQ(a.shape, shape);
  EXPECT_EQ(a.to_vector<T>(), expected);  // exact, including floats

  const fs::path ours = fs::temp_directory_path() / ("sfm_npy_golden_" + name + ".npy");
  npy_write(ours, NpyArray::from(shape, expected));
  EXPECT_EQ(slurp(ours), slurp(numpy_file));
  fs::remove(ours);
}

}  // namespace

// Formulas mirror npy_cases() in tools/make_golden_p1.py.
TEST(NpyGolden, MatchesNumpyForEveryDtype) {
  std::vector<std::uint8_t> u8(24);
  for (std::size_t i = 0; i < u8.size(); ++i) u8[i] = static_cast<std::uint8_t>(i * 11 % 256);
  check_case<std::uint8_t>("u8_2x3x4", {2, 3, 4}, u8);

  std::vector<std::int32_t> i32(12);
  for (std::size_t i = 0; i < i32.size(); ++i) {
    i32[i] = (static_cast<std::int32_t>(i) - 6) * 1000003;
  }
  check_case<std::int32_t>("i32_3x4", {3, 4}, i32);

  check_case<std::uint32_t>("u32_5", {5}, {0u, 1u, 2147483648u, 4294967295u, 7u});

  std::vector<float> f32(6);
  for (std::size_t i = 0; i < f32.size(); ++i) f32[i] = static_cast<float>(i) * 0.5f - 1.25f;
  check_case<float>("f32_2x3", {2, 3}, f32);

  std::vector<double> f64(12);
  for (std::size_t i = 0; i < f64.size(); ++i) f64[i] = static_cast<double>(i) / 7.0 - 0.5;
  check_case<double>("f64_3x4", {3, 4}, f64);

  check_case<double>("f64_scalar", {}, {3.141592653589793});
}

// The P1 data files load with the expected layout (their values are checked by the data tests).
TEST(NpyGolden, P1DataFilesLoad) {
  const NpyArray gray = npy_read(kGolden / "gray_0000.npy");
  EXPECT_EQ(gray.dtype, DType::kU8);
  EXPECT_EQ(gray.shape, (std::vector<std::size_t>{768, 1024}));

  for (const char* name : {"krt_K.npy", "krt_R.npy"}) {
    const NpyArray m = npy_read(kGolden / name);
    EXPECT_EQ(m.dtype, DType::kF64) << name;
    EXPECT_EQ(m.shape, (std::vector<std::size_t>{36, 3, 3})) << name;
  }
  const NpyArray t = npy_read(kGolden / "krt_t.npy");
  EXPECT_EQ(t.shape, (std::vector<std::size_t>{36, 3}));

  // Rotations are orthonormal with det +1 — a cheap check that the bytes are interpreted right.
  const std::vector<double> r = npy_read(kGolden / "krt_R.npy").to_vector<double>();
  for (std::size_t c = 0; c < 36; ++c) {
    const double* m = &r[c * 9];
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    EXPECT_NEAR(det, 1.0, 1e-12) << "camera " << c;
  }
}
