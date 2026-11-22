#include "sfm/core/npy.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "npy I/O assumes a little-endian host");

namespace sfm::core {

namespace {

constexpr std::array<unsigned char, 6> kMagic = {0x93, 'N', 'U', 'M', 'P', 'Y'};
constexpr std::size_t kHeaderAlign = 64;  // np.save pads magic+header to a multiple of 64 bytes

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
  throw IoError("npy " + path.string() + ": " + what);
}

std::size_t checked_count(const std::vector<std::size_t>& shape) {
  std::size_t n = 1;
  for (std::size_t d : shape) {
    if (d != 0 && n > std::numeric_limits<std::size_t>::max() / d) {
      throw IoError("npy: shape product overflows size_t");
    }
    n *= d;
  }
  return n;
}

// Text after "'key':" up to (not including) the next top-level ',' or '}' — or the whole
// parenthesised tuple for 'shape'.
std::string_view field(const std::filesystem::path& path, std::string_view header,
                       std::string_view key) {
  const std::string quoted = "'" + std::string(key) + "'";
  std::size_t pos = header.find(quoted);
  if (pos == std::string_view::npos) fail(path, "header has no " + quoted);
  pos = header.find(':', pos + quoted.size());
  if (pos == std::string_view::npos) fail(path, "header: no ':' after " + quoted);
  ++pos;
  while (pos < header.size() && header[pos] == ' ') ++pos;
  const bool tuple = pos < header.size() && header[pos] == '(';
  const std::size_t end = tuple ? header.find(')', pos) : header.find_first_of(",}", pos);
  if (end == std::string_view::npos) fail(path, "header: unterminated value for " + quoted);
  return header.substr(pos, end - pos + (tuple ? 1 : 0));
}

DType parse_descr(const std::filesystem::path& path, std::string_view v) {
  if (v.size() < 2 || v.front() != '\'' || v.back() != '\'') fail(path, "descr is not a string");
  v = v.substr(1, v.size() - 2);
  if (v == "|u1" || v == "<u1") return DType::kU8;
  if (v == "<i4") return DType::kI32;
  if (v == "<u4") return DType::kU32;
  if (v == "<f4") return DType::kF32;
  if (v == "<f8") return DType::kF64;
  fail(path, "unsupported dtype '" + std::string(v) + "' (want |u1 <i4 <u4 <f4 <f8)");
}

std::vector<std::size_t> parse_shape(const std::filesystem::path& path, std::string_view v) {
  // "()", "(5,)", "(2, 3, 4)"
  std::vector<std::size_t> shape;
  std::size_t i = 1;
  while (i + 1 < v.size()) {
    while (i < v.size() && (v[i] == ' ' || v[i] == ',')) ++i;
    if (i + 1 >= v.size()) break;
    if (!std::isdigit(static_cast<unsigned char>(v[i]))) fail(path, "bad shape " + std::string(v));
    std::size_t d = 0;
    while (i < v.size() && std::isdigit(static_cast<unsigned char>(v[i]))) {
      const auto digit = static_cast<std::size_t>(v[i] - '0');
      if (d > (std::numeric_limits<std::size_t>::max() - digit) / 10) fail(path, "shape overflow");
      d = d * 10 + digit;
      ++i;
    }
    shape.push_back(d);
  }
  return shape;
}

std::string make_header(const NpyArray& a) {
  std::string shape = "(";
  for (std::size_t i = 0; i < a.shape.size(); ++i) {
    shape += (i ? ", " : "") + std::to_string(a.shape[i]);
  }
  shape += a.shape.size() == 1 ? ",)" : ")";
  std::string h =
      "{'descr': '" + to_string(a.dtype) + "', 'fortran_order': False, 'shape': " + shape + ", }";
  const std::size_t prefix = kMagic.size() + 2 + 2;  // magic, version, uint16 header length
  const std::size_t total =
      (prefix + h.size() + 1 + kHeaderAlign - 1) / kHeaderAlign * kHeaderAlign;
  h.append(total - prefix - h.size() - 1, ' ');
  h += '\n';
  return h;
}

}  // namespace

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kU8:
      return 1;
    case DType::kI32:
    case DType::kU32:
    case DType::kF32:
      return 4;
    case DType::kF64:
      return 8;
  }
  throw IoError("dtype_size: invalid DType");
}

std::string to_string(DType dtype) {
  switch (dtype) {
    case DType::kU8:
      return "|u1";
    case DType::kI32:
      return "<i4";
    case DType::kU32:
      return "<u4";
    case DType::kF32:
      return "<f4";
    case DType::kF64:
      return "<f8";
  }
  throw IoError("to_string: invalid DType");
}

std::size_t NpyArray::count() const { return checked_count(shape); }

NpyArray npy_read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "cannot open");
  const std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  if (in.bad()) fail(path, "read error");

  if (file.size() < kMagic.size() + 2 || !std::equal(kMagic.begin(), kMagic.end(), file.begin())) {
    fail(path, "not an .npy file (bad magic)");
  }
  const unsigned major = file[6];
  std::size_t len_bytes = 0;
  if (major == 1) {
    len_bytes = 2;
  } else if (major == 2 || major == 3) {
    len_bytes = 4;
  } else {
    fail(path, "unsupported format version " + std::to_string(major));
  }
  const std::size_t start = 8 + len_bytes;
  if (file.size() < start) fail(path, "truncated header");
  std::size_t header_len = 0;
  for (std::size_t i = 0; i < len_bytes; ++i) header_len |= std::size_t{file[8 + i]} << (8 * i);
  if (file.size() < start + header_len) fail(path, "truncated header");
  const std::string header(file.begin() + static_cast<std::ptrdiff_t>(start),
                           file.begin() + static_cast<std::ptrdiff_t>(start + header_len));

  NpyArray a;
  a.dtype = parse_descr(path, field(path, header, "descr"));
  const std::string_view order = field(path, header, "fortran_order");
  if (order != "False") fail(path, "fortran_order arrays are not supported");
  a.shape = parse_shape(path, field(path, header, "shape"));

  const std::size_t count = a.count();
  if (count > std::numeric_limits<std::size_t>::max() / dtype_size(a.dtype)) {
    fail(path, "data size overflows size_t");
  }
  const std::size_t expected = count * dtype_size(a.dtype);
  const std::size_t actual = file.size() - start - header_len;
  if (actual != expected) {
    fail(path, "data is " + std::to_string(actual) + " bytes, shape and dtype need " +
                   std::to_string(expected));
  }
  a.bytes.assign(file.begin() + static_cast<std::ptrdiff_t>(start + header_len), file.end());
  return a;
}

void npy_write(const std::filesystem::path& path, const NpyArray& array) {
  if (array.bytes.size() != array.count() * dtype_size(array.dtype)) {
    fail(path, "bytes.size() does not match shape and dtype");
  }
  const std::string header = make_header(array);
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) fail(path, "header too long");

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) fail(path, "cannot open for writing");
  out.write(reinterpret_cast<const char*>(kMagic.data()), kMagic.size());
  const std::array<char, 4> version_and_len = {1, 0, static_cast<char>(header.size() & 0xff),
                                               static_cast<char>(header.size() >> 8)};
  out.write(version_and_len.data(), version_and_len.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(array.bytes.data()),
            static_cast<std::streamsize>(array.bytes.size()));
  if (!out) fail(path, "write failed");
}

}  // namespace sfm::core
