#pragma once
// Exception hierarchy of the library. Everything thrown by sfm derives from sfm::Error, so an app
// can catch one type at main(). CudaError (cuda_check.h) is also an sfm::Error.

#include <stdexcept>
#include <string>

namespace sfm {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Invalid or inconsistent run configuration. The message names the JSON path, e.g. "dataset.root".
class ConfigError : public Error {
 public:
  using Error::Error;
};

// File missing, unreadable, or in an unexpected format.
class IoError : public Error {
 public:
  using Error::Error;
};

// Arena out of capacity or invalid allocation request.
class ArenaError : public Error {
 public:
  using Error::Error;
};

// Degenerate geometric input, e.g. collinear points where a rotation is not determined.
class GeometryError : public Error {
 public:
  using Error::Error;
};

// A run stopped on request (e.g. the viewer's Cancel button). Not a failure of the inputs.
class Cancelled : public Error {
 public:
  using Error::Error;
};

}  // namespace sfm
