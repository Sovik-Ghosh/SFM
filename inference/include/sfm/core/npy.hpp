#pragma once
// NumPy .npy read/write: the exchange format between the C++ runtime and the Python tools (golden
// files, debug dumps). Supported: dtypes u8, i32, u32, f32, f64; little-endian; C order; any rank
// (0-d scalar included). Files we write are byte-identical to np.save for the same array.
// Format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sfm/core/error.hpp"

namespace sfm::core {

enum class DType : std::uint8_t { kU8, kI32, kU32, kF32, kF64 };

std::size_t dtype_size(DType dtype);
std::string to_string(DType dtype);  // the .npy descr: "|u1", "<i4", "<u4", "<f4", "<f8"

template <class T>
constexpr DType dtype_of() {
  if constexpr (std::is_same_v<T, std::uint8_t>) return DType::kU8;
  if constexpr (std::is_same_v<T, std::int32_t>) return DType::kI32;
  if constexpr (std::is_same_v<T, std::uint32_t>) return DType::kU32;
  if constexpr (std::is_same_v<T, float>) return DType::kF32;
  if constexpr (std::is_same_v<T, double>) return DType::kF64;
  static_assert(std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int32_t> ||
                    std::is_same_v<T, std::uint32_t> || std::is_same_v<T, float> ||
                    std::is_same_v<T, double>,
                "unsupported npy element type");
}

// An n-d array as raw little-endian bytes in C (row-major) order.
struct NpyArray {
  DType dtype = DType::kU8;
  std::vector<std::size_t> shape;  // empty = 0-d scalar (one element)
  std::vector<unsigned char> bytes;

  std::size_t count() const;  // product of shape (1 for a scalar)

  // Typed copy-in / copy-out. Throws IoError if values.size() != product(shape) or T != dtype.
  template <class T>
  static NpyArray from(std::vector<std::size_t> shape, const std::vector<T>& values) {
    NpyArray a{dtype_of<T>(), std::move(shape), {}};
    if (values.size() != a.count()) throw IoError("NpyArray::from: value count != shape product");
    // assign(), not resize()+memcpy: GCC 13 -O3 reports a bogus -Warray-bounds on the latter.
    const auto* src = reinterpret_cast<const unsigned char*>(values.data());
    a.bytes.assign(src, src + values.size() * sizeof(T));
    return a;
  }

  template <class T>
  std::vector<T> to_vector() const {
    if (dtype_of<T>() != dtype) {
      throw IoError("NpyArray::to_vector: array holds " + to_string(dtype) + ", asked for " +
                    to_string(dtype_of<T>()));
    }
    std::vector<T> out(bytes.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
  }
};

// Throws IoError on: missing/unreadable file, bad magic, unsupported version, dtype, byte order or
// fortran_order, malformed header, or a data section shorter or longer than the shape implies.
NpyArray npy_read(const std::filesystem::path& path);

// Writes format version 1.0. Throws IoError if the file cannot be written or bytes.size() does not
// match shape and dtype.
void npy_write(const std::filesystem::path& path, const NpyArray& array);

}  // namespace sfm::core
