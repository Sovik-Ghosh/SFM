#pragma once
// Portable random numbers for synthetic scenes and RANSAC seeds. std::mt19937_64's bit sequence is
// fixed by the C++ standard, but std::uniform_real_distribution / std::normal_distribution are not
// (libstdc++, libc++ and MSVC give different values). So only the raw bits come from the standard
// engine; the conversions to uniform and normal values are our own and the same everywhere.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace sfm::eval {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  std::uint64_t bits() { return engine_(); }

  // Uniform in [0, 1): the top 53 bits as a double (every value is exactly representable).
  double uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }

  // Uniform in [lo, hi).
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }

  // Uniform integer in [0, n), n > 0. The bias of the floor mapping is below 2^-53 * n.
  std::size_t index(std::size_t n) {
    const auto i = static_cast<std::size_t>(uniform() * static_cast<double>(n));
    return i < n ? i : n - 1;
  }

  // Standard normal (mean 0, variance 1), Box-Muller; values come in pairs, the second is cached.
  // Results depend on the platform's log/sin/cos only in the last bit or so.
  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = 1.0 - uniform();  // (0, 1]: log(u1) is finite
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 2.0 * kPi * u2;
    spare_ = r * std::sin(a);
    has_spare_ = true;
    return r * std::cos(a);
  }

  double normal(double mean, double sigma) { return mean + sigma * normal(); }

 private:
  static constexpr double kPi = 3.14159265358979323846;
  std::mt19937_64 engine_;
  double spare_ = 0.0;
  bool has_spare_ = false;
};

}  // namespace sfm::eval
