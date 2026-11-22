// Host -> device upload bandwidth for the dataset's image stack (P1 design, "Benchmark").
//
//   bench_upload [--mib N] [--runs N] [--warmup N]
//
// Cases: pageable vs pinned host memory, and pinned split over 1 vs 2 streams. Each case runs
// `warmup` untimed copies, then `runs` copies timed with CUDA events; prints min / median / max
// GB/s (1 GB = 1e9 bytes). Uploads cross PCIe, so compare with the link (not device-memory)
// bandwidth. Default size: 36 images x 1024 x 768 gray bytes = 27 MiB, the bunny stack.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "sfm/core/cuda_check.h"
#include "sfm/core/cuda_timer.hpp"
#include "sfm/core/device_buffer.hpp"
#include "sfm/core/device_info.hpp"
#include "sfm/core/pinned_buffer.hpp"
#include "sfm/core/stream.hpp"

namespace {

using sfm::core::CudaTimer;
using sfm::core::DeviceBuffer;
using sfm::core::Event;
using sfm::core::Stream;

struct Options {
  std::size_t bytes = std::size_t{36} * 1024 * 768;
  int runs = 100;
  int warmup = 10;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const long v = std::strtol(argv[i + 1], nullptr, 10);
    if (v <= 0) throw std::runtime_error("bad value for " + key);
    if (key == "--mib") {
      o.bytes = static_cast<std::size_t>(v) << 20;
    } else if (key == "--runs") {
      o.runs = static_cast<int>(v);
    } else if (key == "--warmup") {
      o.warmup = static_cast<int>(v);
    } else {
      throw std::runtime_error("unknown option " + key);
    }
  }
  if (argc % 2 == 0) {
    throw std::runtime_error("usage: bench_upload [--mib N] [--runs N] [--warmup N]");
  }
  return o;
}

// Copies src -> dst split into `parts` equal chunks on `parts` streams; returns milliseconds
// measured on streams[0], which waits for all others before its stop event.
float timed_upload(DeviceBuffer<unsigned char>& dst, const unsigned char* src, std::size_t bytes,
                   std::vector<Stream>& streams) {
  const std::size_t parts = streams.size();
  CudaTimer timer(streams[0]);
  Event begin, end;
  timer.start();
  begin.record(streams[0]);
  for (std::size_t p = 0; p < parts; ++p) {
    if (p > 0) CHECK(cudaStreamWaitEvent(streams[p].get(), begin.get(), 0));
    const std::size_t off = bytes * p / parts;
    const std::size_t len = bytes * (p + 1) / parts - off;
    dst.upload(src + off, len, streams[p], off);
  }
  for (std::size_t p = 1; p < parts; ++p) {  // join: stream 0 waits for every other stream
    end.record(streams[p]);
    CHECK(cudaStreamWaitEvent(streams[0].get(), end.get(), 0));
  }
  timer.stop();
  return timer.elapsed_ms();
}

void run_case(const char* name, const unsigned char* src, const Options& o, int n_streams,
              DeviceBuffer<unsigned char>& dst) {
  std::vector<Stream> streams(static_cast<std::size_t>(n_streams));
  for (int i = 0; i < o.warmup; ++i) timed_upload(dst, src, o.bytes, streams);
  std::vector<double> gbps;
  for (int i = 0; i < o.runs; ++i) {
    const double ms = timed_upload(dst, src, o.bytes, streams);
    gbps.push_back(static_cast<double>(o.bytes) / (ms * 1e6));
  }
  std::sort(gbps.begin(), gbps.end());
  std::printf("%-22s %8.2f %8.2f %8.2f\n", name, gbps.front(), gbps[gbps.size() / 2], gbps.back());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    const auto dev = sfm::core::query_device(0);
    std::printf("device  %s (sm_%d%d)\n", dev.name.c_str(), dev.cc_major, dev.cc_minor);
    std::printf("upload  %.1f MiB host -> device, %d warm-up + %d timed runs per case\n\n",
                static_cast<double>(o.bytes) / (1 << 20), o.warmup, o.runs);

    DeviceBuffer<unsigned char> dst(o.bytes);
    std::vector<unsigned char> pageable(o.bytes);
    sfm::core::PinnedBuffer<unsigned char> pinned(o.bytes);
    for (std::size_t i = 0; i < o.bytes; ++i) pageable[i] = static_cast<unsigned char>(i * 31u);
    std::memcpy(pinned.data(), pageable.data(), o.bytes);

    std::printf("%-22s %8s %8s %8s   (GB/s)\n", "case", "min", "median", "max");
    run_case("pageable, 1 stream", pageable.data(), o, 1, dst);
    run_case("pinned,   1 stream", pinned.data(), o, 1, dst);
    run_case("pinned,   2 streams", pinned.data(), o, 2, dst);

    std::vector<unsigned char> back(o.bytes);  // the copies must have landed intact
    CHECK(cudaMemcpy(back.data(), dst.data(), o.bytes, cudaMemcpyDeviceToHost));
    if (back != pageable) throw std::runtime_error("device data differs from the source");
    std::printf("\ncheck   device copy matches the source\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
