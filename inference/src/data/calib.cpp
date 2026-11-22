#include "sfm/data/calib.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "sfm/core/error.hpp"

namespace sfm::data {

Mat34 read_projection(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw IoError("calib " + path.string() + ": cannot open");

  std::string header;
  std::getline(in, header);
  if (!header.empty() && header.back() == '\r') header.pop_back();  // tolerate CRLF
  if (header != "CONTOUR") {
    throw IoError("calib " + path.string() + ": first line must be CONTOUR, got \"" + header +
                  "\"");
  }

  Mat34 p;
  std::string token;
  int n = 0;
  while (in >> token) {
    if (n == 12) throw IoError("calib " + path.string() + ": more than 12 numbers");
    double v = 0.0;
    std::istringstream conv(token);
    if (!(conv >> v) || !conv.eof() || !std::isfinite(v)) {
      throw IoError("calib " + path.string() + ": not a number: \"" + token + "\"");
    }
    p(n / 4, n % 4) = v;
    ++n;
  }
  if (n != 12) {
    throw IoError("calib " + path.string() + ": expected 12 numbers, got " + std::to_string(n));
  }
  return p;
}

}  // namespace sfm::data
