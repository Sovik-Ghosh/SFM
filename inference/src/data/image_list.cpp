#include "sfm/data/image_list.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>

#include "sfm/core/error.hpp"
#include "sfm/data/calib.hpp"

namespace sfm::data {

namespace fs = std::filesystem;

namespace {

std::string numbered(int id) {
  const std::string s = std::to_string(id);
  return std::string(s.size() < 4 ? 4 - s.size() : 0, '0') + s;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool is_image(const fs::path& p) {
  const std::string e = lower(p.extension().string());
  return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".ppm" || e == ".pgm";
}

bool is_jpeg(const fs::path& p) {
  const std::string e = lower(p.extension().string());
  return e == ".jpg" || e == ".jpeg";
}

// ---- minimal EXIF (TIFF) reader ---------------------------------------------------------------

struct Tiff {
  std::vector<std::uint8_t> b;  // the TIFF block (starts with "II" or "MM")
  bool le = true;

  bool in(std::size_t off, std::size_t n) const { return off <= b.size() && n <= b.size() - off; }
  std::uint32_t byte(std::size_t o) const { return b[o]; }
  std::uint32_t u16(std::size_t o) const {
    if (le) return byte(o) | (byte(o + 1) << 8);
    return (byte(o) << 8) | byte(o + 1);
  }
  std::uint32_t u32(std::size_t o) const {
    if (le) return u16(o) | (u16(o + 2) << 16);
    return (u16(o) << 16) | u16(o + 2);
  }
  // Entry offset of `tag` in the IFD at `ifd`, or 0.
  std::size_t find(std::size_t ifd, std::uint32_t tag) const {
    if (ifd == 0 || !in(ifd, 2)) return 0;
    const std::uint32_t n = u16(ifd);
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::size_t e = ifd + 2 + 12 * static_cast<std::size_t>(i);
      if (!in(e, 12)) return 0;
      if (u16(e) == tag) return e;
    }
    return 0;
  }
  // SHORT or LONG value stored in the entry.
  std::uint32_t integer(std::size_t e) const { return u16(e + 2) == 3 ? u16(e + 8) : u32(e + 8); }
  // RATIONAL value (stored at an offset), 0 when malformed.
  double rational(std::size_t e) const {
    const std::size_t o = u32(e + 8);
    if (u16(e + 2) != 5 || !in(o, 8) || u32(o + 4) == 0) return 0.0;
    return static_cast<double>(u32(o)) / static_cast<double>(u32(o + 4));
  }
};

// The EXIF TIFF block of a JPEG (APP1 "Exif\0\0"), or nullopt.
std::optional<Tiff> read_exif(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::nullopt;
  unsigned char soi[2];
  if (!f.read(reinterpret_cast<char*>(soi), 2) || soi[0] != 0xFF || soi[1] != 0xD8) {
    return std::nullopt;
  }
  for (;;) {
    unsigned char m[4];
    if (!f.read(reinterpret_cast<char*>(m), 4) || m[0] != 0xFF) return std::nullopt;
    if (m[1] == 0xDA || m[1] == 0xD9) return std::nullopt;  // image data: no EXIF before it
    const std::size_t len = (static_cast<std::size_t>(m[2]) << 8) | m[3];
    if (len < 2) return std::nullopt;
    std::vector<std::uint8_t> seg(len - 2);
    if (!f.read(reinterpret_cast<char*>(seg.data()), static_cast<std::streamsize>(seg.size()))) {
      return std::nullopt;
    }
    if (m[1] == 0xE1 && seg.size() > 14 && std::equal(seg.begin(), seg.begin() + 6, "Exif\0\0")) {
      Tiff t;
      t.b.assign(seg.begin() + 6, seg.end());
      if (t.b[0] == 'I' && t.b[1] == 'I') {
        t.le = true;
      } else if (t.b[0] == 'M' && t.b[1] == 'M') {
        t.le = false;
      } else {
        return std::nullopt;
      }
      return t;
    }
  }
}

}  // namespace

std::optional<double> exif_focal_px(const fs::path& jpeg, int max_side, double sensor_width_mm) {
  const auto t = read_exif(jpeg);
  if (!t || !t->in(4, 4)) return std::nullopt;
  const std::size_t ifd0 = t->u32(4);
  const std::size_t ptr = t->find(ifd0, 0x8769);  // Exif sub-IFD
  if (ptr == 0) return std::nullopt;
  const std::size_t exif = t->u32(ptr + 8);
  if (const std::size_t e = t->find(exif, 0xA405); e != 0) {  // FocalLengthIn35mmFilm
    const std::uint32_t f35 = t->integer(e);
    // 35 mm film is 36 mm wide: the longer image side maps to 36 mm.
    if (f35 > 0) return static_cast<double>(f35) / 36.0 * max_side;
  }
  if (sensor_width_mm > 0.0) {
    if (const std::size_t e = t->find(exif, 0x920A); e != 0) {  // FocalLength (mm)
      const double f = t->rational(e);
      if (f > 0.0) return f / sensor_width_mm * max_side;
    }
  }
  return std::nullopt;
}

int exif_orientation(const fs::path& jpeg) {
  const auto t = read_exif(jpeg);
  if (!t || !t->in(4, 4)) return 1;
  const std::size_t e = t->find(t->u32(4), 0x0112);
  if (e == 0) return 1;
  const auto o = static_cast<int>(t->integer(e));
  return o >= 1 && o <= 8 ? o : 1;
}

std::vector<ImageEntry> list_images(const config::DatasetConfig& d) {
  const fs::path images = d.root / d.images_dir;
  const fs::path masks = d.root / d.masks_dir;
  const fs::path calib = d.root / d.calib_dir;
  const bool shared = d.camera.enabled();
  std::vector<ImageEntry> out;

  if (d.layout != "folder") {
    if (d.first < 0 || d.last < d.first || d.last > 9999) {
      throw IoError("dataset: invalid id range " + std::to_string(d.first) + ".." +
                    std::to_string(d.last));
    }
    for (int id = d.first; id <= d.last; ++id) {
      const std::string n = numbered(id);
      ImageEntry e;
      e.id = id;
      e.name = n + ".ppm";
      e.image = images / e.name;
      if (d.use_masks) e.mask = masks / (n + ".pgm");
      if (!shared) e.calib = calib / (n + ".txt");
      out.push_back(std::move(e));
    }
    return out;
  }

  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto& it : fs::directory_iterator(images, ec)) {
    if (it.is_regular_file() && is_image(it.path())) files.push_back(it.path());
  }
  if (ec) throw IoError("dataset: cannot list " + images.string() + ": " + ec.message());
  std::sort(files.begin(), files.end(),
            [](const fs::path& a, const fs::path& b) { return a.filename() < b.filename(); });
  if (files.empty()) {
    throw IoError("dataset: no images (.jpg .jpeg .png .ppm .pgm) in " + images.string());
  }
  const int count = static_cast<int>(files.size());
  const int last = d.last < 0 ? count - 1 : d.last;
  if (d.first < 0 || d.first > last || last >= count) {
    throw IoError("dataset: range " + std::to_string(d.first) + ".." + std::to_string(d.last) +
                  " does not fit the " + std::to_string(count) + " images in " + images.string() +
                  " (indices 0.." + std::to_string(count - 1) + ")");
  }
  for (int i = d.first; i <= last; ++i) {
    const fs::path& f = files[static_cast<std::size_t>(i)];
    ImageEntry e;
    e.id = i;
    e.name = f.filename().string();
    e.image = f;
    const std::string stem = f.stem().string();
    if (d.use_masks) {
      for (const char* ext : {".png", ".pgm", ".PNG", ".PGM"}) {
        if (fs::exists(masks / (stem + ext))) {
          e.mask = masks / (stem + ext);
          break;
        }
      }
      if (e.mask.empty()) {
        throw IoError("dataset: no mask for " + e.name + " (" + (masks / (stem + ".png")).string() +
                      " or .pgm)");
      }
    }
    if (!shared) e.calib = calib / (stem + ".txt");
    out.push_back(std::move(e));
  }
  return out;
}

Krt load_camera(const config::DatasetConfig& d, const ImageEntry& e, int width, int height) {
  if (!d.camera.enabled()) return decompose_projection(read_projection(e.calib));
  const config::CameraConfig& c = d.camera;
  const int max_side = std::max(width, height);
  double fx = c.fx;
  if (c.exif) {
    std::optional<double> f;
    if (is_jpeg(e.image)) f = exif_focal_px(e.image, max_side, c.sensor_width_mm);
    fx = f ? *f : c.fx_default_factor * max_side;
  }
  const double fy = c.fy > 0.0 ? c.fy : fx;
  const double cx = c.cx >= 0.0 ? c.cx : 0.5 * (width - 1);
  const double cy = c.cy >= 0.0 ? c.cy : 0.5 * (height - 1);
  Krt k;
  k.K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
  k.R.setIdentity();
  k.t.setZero();
  return k;
}

}  // namespace sfm::data
