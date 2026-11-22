#pragma once
// Runs a reconstruction on a worker thread for the viewer. The worker calls a Runner (normally
// run_pipeline with a progress sink); progress arrives as copies (snapshots + events) in a queue
// that the UI thread drains with poll(). The worker never touches OpenGL or ImGui.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/progress.hpp"

namespace sfm::viewer {

enum class RunState : std::uint8_t { kIdle, kRunning, kFinished, kCancelled, kFailed };

enum class EventKind : std::uint8_t {
  kStageBegin,
  kStageEnd,
  kCamera,            // initial pair or a newly registered camera (has a snapshot)
  kBundleAdjustment,  // after a BA and the filtering that follows it (has a snapshot)
  kFinished,
  kCancelled,
  kFailed,
};

struct RunEvent {
  EventKind kind = EventKind::kStageBegin;
  pipeline::Stage stage = pipeline::Stage::kFeatures;  // stage events
  double seconds = 0.0;   // stage end: stage time; others: time since start()
  int snapshot = -1;      // camera / BA: index of the snapshot in this run (== its sequence)
  int cameras = 0;        // camera / BA: registered cameras in that snapshot
  int points = 0;         // camera / BA: points in that snapshot
  int newest_image = -1;  // camera: image index registered last
  std::string message;    // failed: the error text
};

// Everything that arrived since the previous poll().
struct RunUpdate {
  std::vector<pipeline::SceneSnapshot> snapshots;  // in order, consecutive sequence numbers
  std::vector<RunEvent> events;                    // in order
  RunState state = RunState::kIdle;
  std::optional<pipeline::Metrics> metrics;  // set once, with the kFinished event
  std::optional<pipeline::Timings> timings;
};

class RunController {
 public:
  // Runs the reconstruction, reporting to `sink`. Returns normally on success; throws
  // sfm::Cancelled when sink.cancelled() turned true, anything else on failure.
  using Runner = std::function<void(pipeline::ProgressSink& sink)>;

  RunController() = default;
  ~RunController();  // cancels a running run and waits for the worker
  RunController(const RunController&) = delete;
  RunController& operator=(const RunController&) = delete;

  // Starts `runner` on a new worker thread. Returns false (and does nothing) while a run is active.
  bool start(Runner runner);
  // Asks the run to stop at its next check (between stages, images and registrations).
  void cancel();
  bool active() const { return state() == RunState::kRunning; }
  RunState state() const;
  double elapsed_seconds() const;  // since start(); frozen when the run ends
  // Moves out what arrived since the last call. Joins the worker once the run has ended.
  RunUpdate poll();

 private:
  class Sink;
  void push_event(RunEvent e);
  void push_snapshot(const pipeline::SceneSnapshot& s, EventKind kind);
  void finish(RunState state, std::string message);
  void join();

  mutable std::mutex mutex_;
  RunUpdate pending_;
  RunState state_ = RunState::kIdle;
  std::atomic<bool> cancel_{false};
  std::chrono::steady_clock::time_point started_{};
  std::chrono::steady_clock::time_point ended_{};
  std::thread worker_;
};

// Which implementation the run uses.
enum class BackendChoice : std::uint8_t { kAuto, kGpu, kCpu };

// cfg with every method set for `choice`: kAuto resolves "auto" (GPU methods if `gpu_available`),
// kGpu maps the CPU method names to their CUDA versions (orb/sift/bf/lm_schur _cpu -> _cuda),
// kCpu maps CUDA names to CPU ones. Other fields are unchanged.
config::Config with_backend(const config::Config& cfg, BackendChoice choice, bool gpu_available);

// The real runner: run_pipeline(cfg) with the sink (log lines go to <run_dir>/run.log only).
RunController::Runner pipeline_runner(config::Config cfg);

}  // namespace sfm::viewer
