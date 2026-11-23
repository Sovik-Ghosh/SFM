// Python bindings: epipole._epipole. The Python package (python/epipole/__init__.py) wraps these
// in run() / reconstruct() and a Result class; see there for the documented API.
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "default_config.hpp"
#include "sfm/config/config.hpp"
#include "sfm/core/device_info.hpp"
#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/core/version.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/registry.hpp"

namespace py = pybind11;
namespace fs = std::filesystem;

namespace {

// Same path rules as sfm::config::load_config: relative paths against base_dir.
sfm::config::Config load(const std::string& json_text, const std::string& base_dir) {
  sfm::config::Config c = sfm::config::parse_config(json_text);
  const fs::path base(base_dir);
  const auto absolute = [&](const fs::path& p) {
    return (p.is_absolute() ? p : base / p).lexically_normal();
  };
  if (!c.run.output_dir.empty()) c.run.output_dir = absolute(c.run.output_dir);
  if (!c.dataset.root.empty()) c.dataset.root = absolute(c.dataset.root);
  sfm::config::validate_config(c, base);
  return c;
}

py::array_t<double> matrices(const std::vector<Eigen::Matrix3d>& v) {
  py::array_t<double> a({static_cast<py::ssize_t>(v.size()), py::ssize_t{3}, py::ssize_t{3}});
  auto m = a.mutable_unchecked<3>();
  for (std::size_t i = 0; i < v.size(); ++i) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m(static_cast<py::ssize_t>(i), r, c) = v[i](r, c);
    }
  }
  return a;
}

py::array_t<double> vectors(const std::vector<Eigen::Vector3d>& v) {
  py::array_t<double> a({static_cast<py::ssize_t>(v.size()), py::ssize_t{3}});
  auto m = a.mutable_unchecked<2>();
  for (std::size_t i = 0; i < v.size(); ++i) {
    for (int c = 0; c < 3; ++c) m(static_cast<py::ssize_t>(i), c) = v[i](c);
  }
  return a;
}

// Runs a config (JSON text). Returns a dict of plain values and NumPy arrays.
py::dict run(const std::string& json_text, const std::string& base_dir, const std::string& backend,
             const std::string& log_level) {
  sfm::config::Config cfg = load(json_text, base_dir);
  if (backend != "auto") {
    if (backend != "cpu" && backend != "gpu") {
      throw sfm::ConfigError("backend: expected \"auto\", \"cpu\" or \"gpu\", got \"" + backend +
                             "\"");
    }
    cfg = sfm::pipeline::use_backend(cfg, backend == "gpu");
  }
  sfm::pipeline::RunResult r;
  {
    py::gil_scoped_release release;  // the run takes seconds to minutes
    sfm::core::Logger log(sfm::core::parse_log_level(log_level), &std::cerr);
    r = sfm::pipeline::run_pipeline(cfg, {}, log);
  }
  py::dict out;
  out["run_dir"] = r.run_dir.string();
  out["timings"] = sfm::pipeline::to_json(r.timings);
  out["metrics"] = r.metrics ? py::object(py::str(sfm::pipeline::to_json(*r.metrics))) : py::none();
  if (r.model) {
    const auto& m = *r.model;
    out["names"] = m.names;
    py::array_t<int> sz({static_cast<py::ssize_t>(m.sizes.size()), py::ssize_t{2}});
    for (std::size_t i = 0; i < m.sizes.size(); ++i) {
      sz.mutable_at(static_cast<py::ssize_t>(i), 0) = m.sizes[i][0];
      sz.mutable_at(static_cast<py::ssize_t>(i), 1) = m.sizes[i][1];
    }
    out["sizes"] = sz;
    py::array_t<bool> reg(static_cast<py::ssize_t>(m.registered.size()));
    for (std::size_t i = 0; i < m.registered.size(); ++i) {
      reg.mutable_at(static_cast<py::ssize_t>(i)) = m.registered[i];
    }
    out["registered"] = reg;
    out["K"] = matrices(m.K);
    out["R"] = matrices(m.R);
    out["t"] = vectors(m.t);
    out["points"] = vectors(m.points);
    py::array_t<std::uint8_t> col({static_cast<py::ssize_t>(m.colours.size()), py::ssize_t{3}});
    auto cm = col.mutable_unchecked<2>();
    for (std::size_t i = 0; i < m.colours.size(); ++i) {
      for (int c = 0; c < 3; ++c) cm(static_cast<py::ssize_t>(i), c) = m.colours[i][c];
    }
    out["colors"] = col;
    out["k1"] = m.k1;
    out["k2"] = m.k2;
  }
  return out;
}

py::dict device() {
  const sfm::pipeline::Backend b = sfm::pipeline::detect_backend();
  py::dict d;
  d["cuda_build"] = sfm::pipeline::built_with_cuda();
  d["gpu"] = b.gpu;
  d["device"] = b.device;
  d["unusable"] = b.unusable;
  return d;
}

}  // namespace

PYBIND11_MODULE(_epipole, m) {
  m.doc() = "Epipole: Structure-from-Motion in C++ and CUDA (native module)";
  m.attr("__version__") = sfm::core::version_string();
  m.attr("default_config") = epipole::kDefaultConfig;
  m.def("run", &run, py::arg("json_text"), py::arg("base_dir"), py::arg("backend") = "auto",
        py::arg("log_level") = "warn");
  m.def("device", &device);
  // Translators are tried newest first: the base class is registered first, so it is tried last.
  py::register_exception<sfm::Error>(m, "Error", PyExc_RuntimeError);
  py::register_exception<sfm::IoError>(m, "IoError", PyExc_OSError);
  py::register_exception<sfm::ConfigError>(m, "ConfigError", PyExc_ValueError);
}
