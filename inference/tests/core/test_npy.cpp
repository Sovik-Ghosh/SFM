#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "sfm/core/error.hpp"
#include "sfm/core/npy.hpp"

namespace fs = std::filesystem;
using sfm::IoError;
using sfm::core::DType;
using sfm::core::npy_read;
using sfm::core::npy_write;
using sfm::core::NpyArray;

namespace {

class Npy : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::temp_directory_path() / (std::string("sfm_npy_") + info->name());
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path file(const std::string& name) const { return dir_ / name; }

  void write_raw(const fs::path& p, const std::string& content) const {
    std::ofstream(p, std::ios::binary) << content;
  }

  template <class T>
  void round_trip(const std::vector<std::size_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) n *= d;
    std::vector<T> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = static_cast<T>(i * 7 + 3);
    const auto path = file("rt.npy");
    npy_write(path, NpyArray::from(shape, values));
    const NpyArray back = npy_read(path);
    EXPECT_EQ(back.dtype, sfm::core::dtype_of<T>());
    EXPECT_EQ(back.shape, shape);
    EXPECT_EQ(back.to_vector<T>(), values);
  }

 private:
  fs::path dir_;
};

// Header in the exact np.save layout, data appended.
std::string npy_bytes(const std::string& dict, const std::string& data, char major = 1) {
  std::string h = dict;
  const std::size_t prefix = major == 1 ? 10 : 12;
  while ((prefix + h.size() + 1) % 64 != 0) h += ' ';
  h += '\n';
  std::string out = "\x93NUMPY";
  out += major;
  out += '\0';
  out += static_cast<char>(h.size() & 0xff);
  out += static_cast<char>(h.size() >> 8);
  if (major != 1) out += std::string(2, '\0');
  return out + h + data;
}

}  // namespace

TEST_F(Npy, RoundTripsEveryDtypeAndRank) {
  for (const std::vector<std::size_t>& shape :
       {std::vector<std::size_t>{}, {5}, {3, 4}, {2, 3, 4}, {0}, {2, 0, 3}}) {
    SCOPED_TRACE(shape.size());
    round_trip<std::uint8_t>(shape);
    round_trip<std::int32_t>(shape);
    round_trip<std::uint32_t>(shape);
    round_trip<float>(shape);
    round_trip<double>(shape);
  }
}

TEST_F(Npy, HeaderIsNumpyLayoutAligned) {
  npy_write(file("a.npy"), NpyArray::from<double>({3, 4}, std::vector<double>(12, 1.0)));
  std::ifstream in(file("a.npy"), std::ios::binary);
  const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  ASSERT_EQ(raw.size(), 128u + 12 * 8);  // 128-byte header (multiple of 64) + data
  EXPECT_EQ(raw.substr(0, 10), std::string("\x93NUMPY\x01\x00\x76\x00", 10));  // v1.0, len 118
  const std::string header = raw.substr(10, 118);
  EXPECT_EQ(header.rfind("{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }", 0), 0u);
  EXPECT_EQ(header.back(), '\n');
}

TEST_F(Npy, ReadsVersion2Header) {
  const std::vector<std::int32_t> v = {-1, 2, 3};
  std::string data(reinterpret_cast<const char*>(v.data()), v.size() * 4);
  write_raw(file("v2.npy"),
            npy_bytes("{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }", data, 2));
  EXPECT_EQ(npy_read(file("v2.npy")).to_vector<std::int32_t>(), v);
}

TEST_F(Npy, RejectsBadFiles) {
  const std::string f8x2(16, '\0');
  const std::string ok = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"magic", "\x93NUMPX" + npy_bytes(ok, f8x2).substr(6)},
      {"empty", ""},
      {"version", npy_bytes(ok, f8x2, 9)},
      {"big_endian", npy_bytes("{'descr': '>f8', 'fortran_order': False, 'shape': (2,), }", f8x2)},
      {"complex", npy_bytes("{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }", f8x2)},
      {"fortran", npy_bytes("{'descr': '<f8', 'fortran_order': True, 'shape': (2,), }", f8x2)},
      {"no_shape", npy_bytes("{'descr': '<f8', 'fortran_order': False, }", f8x2)},
      {"bad_shape", npy_bytes("{'descr': '<f8', 'fortran_order': False, 'shape': (x,), }", f8x2)},
      {"truncated_data", npy_bytes(ok, f8x2.substr(0, 15))},
      {"extra_data", npy_bytes(ok, f8x2 + "!")},
      {"truncated_header", npy_bytes(ok, f8x2).substr(0, 40)},
  };
  for (const auto& [name, bytes] : cases) {
    SCOPED_TRACE(name);
    write_raw(file(name + ".npy"), bytes);
    EXPECT_THROW(npy_read(file(name + ".npy")), IoError);
  }
  EXPECT_THROW(npy_read(file("missing.npy")), IoError);
}

TEST_F(Npy, TypedAccessChecksDtypeAndCount) {
  const NpyArray a = NpyArray::from<float>({2}, {1.f, 2.f});
  EXPECT_THROW(a.to_vector<double>(), IoError);
  EXPECT_THROW((NpyArray::from<float>({3}, {1.f, 2.f})), IoError);
}

TEST_F(Npy, WriteRejectsInconsistentArray) {
  NpyArray a = NpyArray::from<double>({2}, {1.0, 2.0});
  a.shape = {3};
  EXPECT_THROW(npy_write(file("bad.npy"), a), IoError);
  EXPECT_THROW(npy_write(file("no_such_dir/x.npy"), NpyArray::from<double>({1}, {1.0})), IoError);
}
