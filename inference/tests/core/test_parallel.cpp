#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "sfm/core/parallel.hpp"

TEST(ParallelFor, EveryIndexOnceForAnyThreadCount) {
  for (const unsigned int threads : {0U, 1U, 3U, 64U}) {
    std::vector<int> hits(1000, 0);
    sfm::core::parallel_for(
        hits.size(), [&](std::size_t i) { hits[i] += static_cast<int>(i); }, threads);
    for (std::size_t i = 0; i < hits.size(); ++i) EXPECT_EQ(hits[i], static_cast<int>(i));
  }
  int calls = 0;
  sfm::core::parallel_for(0, [&](std::size_t) { ++calls; });
  EXPECT_EQ(calls, 0);
  EXPECT_GE(sfm::core::default_threads(), 1U);
}

TEST(ParallelFor, RethrowsAfterJoining) {
  std::atomic<int> done{0};
  EXPECT_THROW(sfm::core::parallel_for(
                   100,
                   [&](std::size_t i) {
                     if (i == 7) throw std::runtime_error("x");
                     ++done;
                   },
                   4),
               std::runtime_error);
  EXPECT_LT(done.load(), 100);
}
