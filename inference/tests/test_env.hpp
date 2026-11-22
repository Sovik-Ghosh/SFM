#pragma once
// Helpers for tests that measure time or memory: those numbers are meaningless under
// compute-sanitizer (instrumentation adds latency and its own device allocations).
// scripts/ci/sanitize.sh exports SFM_UNDER_SANITIZER=1.

#include <cstdlib>

namespace sfm::test {

inline bool under_sanitizer() { return std::getenv("SFM_UNDER_SANITIZER") != nullptr; }

// Input size for GPU parity tests: the full size normally, a small one under compute-sanitizer.
// The sanitizer looks for memory / race / init errors in the kernels, which a small input reaches
// the same way; the full-size results are checked against the CPU in the normal test run.
// (Full-size parity under racecheck took 5-11 min per test and dominated the GPU CI job.)
inline int sanitizer_size(int full, int small) { return under_sanitizer() ? small : full; }

}  // namespace sfm::test

#define SFM_SKIP_UNDER_SANITIZER()                                 \
  do {                                                             \
    if (::sfm::test::under_sanitizer())                            \
      GTEST_SKIP() << "measurement test: skipped under sanitizer"; \
  } while (0)
