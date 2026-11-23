#include <cstdio>
#include <fstream>

#include "sfm/core/error.hpp"
#include "sfm/export/export.hpp"

namespace sfm::exporter {

std::vector<Rgb> gray_to_rgb(const std::vector<std::uint8_t>& gray) {
  std::vector<Rgb> out;
  out.reserve(gray.size());
  for (const std::uint8_t g : gray) out.push_back({g, g, g});
  return out;
}

void write_ply(const std::filesystem::path& file, const recon::Reconstruction& rec,
               const std::vector<std::uint8_t>& gray) {
  write_ply(file, rec, gray_to_rgb(gray));
}

void write_ply(const std::filesystem::path& file, const recon::Reconstruction& rec,
               const std::vector<Rgb>& colours) {
  std::ofstream out(file);
  if (!out) throw IoError("write_ply: cannot open " + file.string());
  out << "ply\nformat ascii 1.0\nelement vertex " << rec.point_count()
      << "\nproperty double x\nproperty double y\nproperty double z\n"
         "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  char line[160];
  for (std::size_t t = 0; t < rec.points.size(); ++t) {
    if (rec.has_point[t] == 0) continue;
    const auto& X = rec.points[t];
    const Rgb c = t < colours.size() ? colours[t] : Rgb{128, 128, 128};
    std::snprintf(line, sizeof line, "%.9g %.9g %.9g %d %d %d\n", X.x(), X.y(), X.z(), c[0], c[1],
                  c[2]);
    out << line;
  }
  if (!out) throw IoError("write_ply: write failed " + file.string());
}

}  // namespace sfm::exporter
