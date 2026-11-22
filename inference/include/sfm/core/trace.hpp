// Named ranges for profiler timelines (NVTX, shown by Nsight Systems: `nsys profile
// --trace=cuda,nvtx`). Costs a few ns per range when no profiler is attached; compiles to nothing
// without NVTX headers.
#pragma once

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define SFM_HAVE_NVTX 1
#endif

namespace sfm::core {

inline void trace_push([[maybe_unused]] const char* name) {
#ifdef SFM_HAVE_NVTX
  nvtxRangePushA(name);
#endif
}

inline void trace_pop() {
#ifdef SFM_HAVE_NVTX
  nvtxRangePop();
#endif
}

// One range for the lifetime of the object.
class TraceRange {
 public:
  explicit TraceRange(const char* name) { trace_push(name); }
  TraceRange(const TraceRange&) = delete;
  TraceRange& operator=(const TraceRange&) = delete;
  TraceRange(TraceRange&&) = delete;
  TraceRange& operator=(TraceRange&&) = delete;
  ~TraceRange() { trace_pop(); }
};

}  // namespace sfm::core
