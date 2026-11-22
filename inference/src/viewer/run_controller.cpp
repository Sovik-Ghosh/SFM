#include "sfm/viewer/run_controller.hpp"

#include <exception>
#include <sstream>
#include <utility>

#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"
#include "sfm/pipeline/registry.hpp"

namespace sfm::viewer {

// Forwards pipeline progress into the controller's queue (runs on the worker thread).
class RunController::Sink : public pipeline::ProgressSink {
 public:
  explicit Sink(RunController& c) : c_(c) {}
  void stage_begin(pipeline::Stage s) override {
    RunEvent e;
    e.kind = EventKind::kStageBegin;
    e.stage = s;
    c_.push_event(std::move(e));
  }
  void stage_end(pipeline::Stage s, double seconds) override {
    RunEvent e;
    e.kind = EventKind::kStageEnd;
    e.stage = s;
    e.seconds = seconds;
    c_.push_event(std::move(e));
  }
  void camera_registered(const pipeline::SceneSnapshot& s) override {
    c_.push_snapshot(s, EventKind::kCamera);
  }
  void ba_done(const pipeline::SceneSnapshot& s, const ba::Report& /*r*/) override {
    c_.push_snapshot(s, EventKind::kBundleAdjustment);
  }
  void finished(const pipeline::Metrics& m, const pipeline::Timings& t) override {
    const std::lock_guard<std::mutex> lock(c_.mutex_);
    c_.pending_.metrics = m;
    c_.pending_.timings = t;
  }
  bool cancelled() const override { return c_.cancel_.load(); }

 private:
  RunController& c_;
};

RunController::~RunController() {
  cancel();
  join();
}

bool RunController::start(Runner runner) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == RunState::kRunning) return false;
  }
  join();  // a previous, finished worker
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    pending_ = RunUpdate{};
    state_ = RunState::kRunning;
    started_ = std::chrono::steady_clock::now();
    ended_ = started_;
  }
  cancel_.store(false);
  worker_ = std::thread([this, r = std::move(runner)] {
    Sink sink(*this);
    try {
      r(sink);
      finish(RunState::kFinished, {});
    } catch (const sfm::Cancelled&) {
      finish(RunState::kCancelled, {});
    } catch (const std::exception& e) {
      finish(RunState::kFailed, e.what());
    }
  });
  return true;
}

void RunController::cancel() { cancel_.store(true); }

RunState RunController::state() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

double RunController::elapsed_seconds() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto end = state_ == RunState::kRunning ? std::chrono::steady_clock::now() : ended_;
  return std::chrono::duration<double>(end - started_).count();
}

RunUpdate RunController::poll() {
  RunUpdate out;
  bool ended = false;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    out = std::move(pending_);
    pending_ = RunUpdate{};
    out.state = state_;
    ended = state_ != RunState::kRunning && state_ != RunState::kIdle;
  }
  if (ended) join();
  return out;
}

void RunController::push_event(RunEvent e) {
  const std::lock_guard<std::mutex> lock(mutex_);
  e.seconds =
      e.kind == EventKind::kStageEnd
          ? e.seconds
          : std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
  pending_.events.push_back(std::move(e));
}

void RunController::push_snapshot(const pipeline::SceneSnapshot& s, EventKind kind) {
  RunEvent e;
  e.kind = kind;
  e.snapshot = s.sequence;
  e.cameras = static_cast<int>(s.order.size());
  e.points = static_cast<int>(s.points.size());
  if (kind == EventKind::kCamera && !s.order.empty()) e.newest_image = s.order.back();
  pipeline::SceneSnapshot copy = s;  // copied outside the lock
  const std::lock_guard<std::mutex> lock(mutex_);
  e.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
  pending_.snapshots.push_back(std::move(copy));
  pending_.events.push_back(std::move(e));
}

void RunController::finish(RunState state, std::string message) {
  RunEvent e;
  e.kind = state == RunState::kFinished    ? EventKind::kFinished
           : state == RunState::kCancelled ? EventKind::kCancelled
                                           : EventKind::kFailed;
  e.message = std::move(message);
  const std::lock_guard<std::mutex> lock(mutex_);
  ended_ = std::chrono::steady_clock::now();
  e.seconds = std::chrono::duration<double>(ended_ - started_).count();
  pending_.events.push_back(std::move(e));
  state_ = state;
}

void RunController::join() {
  if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id()) worker_.join();
}

namespace {

std::string swap_suffix(const std::string& method, const std::string& from, const std::string& to) {
  if (method.size() > from.size() &&
      method.compare(method.size() - from.size(), from.size(), from) == 0) {
    return method.substr(0, method.size() - from.size()) + to;
  }
  return method;
}

}  // namespace

config::Config with_backend(const config::Config& cfg, BackendChoice choice, bool gpu_available) {
  const bool gpu =
      choice == BackendChoice::kGpu || (choice == BackendChoice::kAuto && gpu_available);
  config::Config c = pipeline::resolve_auto_methods(cfg, gpu);
  if (choice == BackendChoice::kAuto) return c;
  const std::string from = gpu ? "_cpu" : "_cuda";
  const std::string to = gpu ? "_cuda" : "_cpu";
  c.features.method = swap_suffix(c.features.method, from, to);
  c.matching.method = swap_suffix(c.matching.method, from, to);
  c.ba.method = swap_suffix(c.ba.method, from, to);
  return c;
}

RunController::Runner pipeline_runner(config::Config cfg) {
  return [cfg = std::move(cfg)](pipeline::ProgressSink& sink) {
    std::ostringstream discard;  // run_pipeline also writes <run_dir>/run.log
    core::Logger log(cfg.run.log_level, &discard, false);
    pipeline::run_pipeline(cfg, {}, log, pipeline::default_registry(), &sink);
  };
}

}  // namespace sfm::viewer
