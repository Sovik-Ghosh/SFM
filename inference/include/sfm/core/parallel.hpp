#pragma once
// parallel_for(n, fn): calls fn(i) for i = 0 .. n - 1 on up to `threads` std::threads (0 = all
// hardware threads). Indices are handed out one at a time from an atomic counter, so uneven work
// balances; fn must only write state owned by index i, which makes the result independent of
// the thread count and the schedule. The first exception thrown by fn is rethrown after all
// threads have joined (the remaining indices are skipped).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace sfm::core {

inline unsigned int default_threads() {
  const unsigned int h = std::thread::hardware_concurrency();
  return h == 0 ? 1U : h;
}

template <class Fn>
void parallel_for(std::size_t n, Fn&& fn, unsigned int threads = 0) {
  if (n == 0) return;
  const auto workers = static_cast<std::size_t>(
      std::min<std::size_t>(n, threads == 0 ? default_threads() : threads));
  if (workers <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto work = [&] {
    for (;;) {
      const std::size_t i = next.fetch_add(1);
      if (i >= n || failed.load()) return;
      try {
        fn(i);
      } catch (...) {
        const std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
        failed.store(true);
      }
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (std::size_t t = 1; t < workers; ++t) pool.emplace_back(work);
  work();
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace sfm::core
