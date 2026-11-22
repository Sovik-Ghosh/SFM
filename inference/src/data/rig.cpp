#include "sfm/data/rig.hpp"

#include <fstream>
#include <map>
#include <sstream>

#include <Eigen/LU>

#include "sfm/core/error.hpp"

namespace sfm::data {

Rig parse_rig(const std::string& text, const std::vector<int>& ids, const std::string& name) {
  std::map<int, Eigen::Matrix<double, 12, 1>> sensors;
  std::map<int, std::pair<int, int>> images;  // id -> (frame, sensor)
  std::istringstream in(text);
  std::string line;
  int line_no = 0;
  const auto bad = [&](const std::string& what) {
    throw IoError(name + ":" + std::to_string(line_no) + ": " + what);
  };
  while (std::getline(in, line)) {
    ++line_no;
    std::istringstream ls(line);
    std::string kind;
    if (!(ls >> kind) || kind[0] == '#') continue;
    if (kind == "SENSOR") {
      int s = -1;
      std::string sensor_name;
      Eigen::Matrix<double, 12, 1> v;
      if (!(ls >> s >> sensor_name) || s < 0) bad("expected SENSOR <s> <name> R(9) t(3)");
      for (int k = 0; k < 12; ++k) {
        if (!(ls >> v(k))) bad("expected SENSOR <s> <name> R(9) t(3)");
      }
      if (!sensors.emplace(s, v).second) bad("sensor " + std::to_string(s) + " given twice");
    } else if (kind == "IMAGE") {
      int id = -1;
      int f = -1;
      int s = -1;
      if (!(ls >> id >> f >> s) || f < 0 || s < 0) bad("expected IMAGE <id> <frame> <sensor>");
      images[id] = {f, s};
    } else {
      bad("unknown line '" + kind + "'");
    }
  }
  Rig rig;
  const int n_sensors = sensors.empty() ? 0 : sensors.rbegin()->first + 1;
  if (static_cast<int>(sensors.size()) != n_sensors) {
    throw IoError(name + ": sensors must be 0..n-1");
  }
  for (const auto& [s, v] : sensors) {
    Eigen::Matrix3d R;
    R << v(0), v(1), v(2), v(3), v(4), v(5), v(6), v(7), v(8);
    if ((R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 1e-6 || R.determinant() < 0) {
      throw IoError(name + ": sensor " + std::to_string(s) + " has no rotation matrix");
    }
    rig.R.push_back(R);
    rig.t.emplace_back(v(9), v(10), v(11));
  }
  std::map<int, int> frame_of;  // file frame -> compact frame
  for (const int id : ids) {
    const auto it = images.find(id);
    if (it == images.end()) throw IoError(name + ": no IMAGE line for image " + std::to_string(id));
    const auto [f, s] = it->second;
    if (s >= n_sensors) throw IoError(name + ": image " + std::to_string(id) + ": unknown sensor");
    const auto fit = frame_of.emplace(f, static_cast<int>(frame_of.size())).first;
    rig.frame.push_back(fit->second);
    rig.sensor.push_back(s);
  }
  rig.frames = static_cast<int>(frame_of.size());
  return rig;
}

Rig load_rig(const std::filesystem::path& file, const std::vector<int>& ids) {
  std::ifstream f(file);
  if (!f) throw IoError(file.string() + ": cannot open (reconstruction.rig needs rig.txt)");
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse_rig(ss.str(), ids, file.string());
}

}  // namespace sfm::data
