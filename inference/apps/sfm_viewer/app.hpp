#pragma once
// The viewer's state and UI (the approved design: header, left rail, 3D viewport, inspector,
// timeline). Runs reconstructions live on a worker thread (RunController) or loads finished runs.
// The window / loop is in main.cpp.

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sfm/config/config.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/registry.hpp"
#include "sfm/viewer/compare.hpp"
#include "sfm/viewer/ground_truth.hpp"
#include "sfm/viewer/history.hpp"
#include "sfm/viewer/layout.hpp"
#include "sfm/viewer/orbit_camera.hpp"
#include "sfm/viewer/run_controller.hpp"
#include "sfm/viewer/run_loader.hpp"
#include "sfm/viewer/scene.hpp"

#include "interop.hpp"
#include "renderer.hpp"

struct GLFWwindow;

namespace sfm::viewer {

class App {
 public:
  explicit App(GLFWwindow* window);
  ~App();
  App(const App&) = delete;
  App& operator=(const App&) = delete;

  // Loads outputs/<run>/; on failure keeps the current scene and shows the error.
  bool load(const std::filesystem::path& run_dir);
  // Reads a config for a live run (shown in the run setup panel). Returns false on an error.
  bool open_config(const std::filesystem::path& file);
  // Starts the opened config on the worker thread with the chosen backend.
  bool start_run();
  void set_backend(BackendChoice b) { backend_ = b; }
  // Compare mode: the current run (A, a finished run on disk) against the run in `b_dir`.
  bool open_compare(const std::filesystem::path& b_dir);
  // Leaves Compare; after a swap the viewer's run is the other one, so it is loaded into the view.
  void close_compare() {
    cmp_.reset();
    if (!run_dir_.empty() && run_dir_ != shown_dir_) load(run_dir_);
  }
  // 0 side by side, 1 overlay, 2 wipe (scripted screenshots).
  void set_compare_layout(int layout);
  bool comparing() const { return cmp_ != nullptr; }
  // Changes the point colouring as the Colour-by control does (GPU recolour when available).
  void set_color_mode(int mode) {
    options_.color = static_cast<ColorMode>(mode);
    recolor_ = true;
  }
  bool gpu_colors() const { return gpu_colors_ != nullptr; }
  // In-app screenshots: the UI asks (button / S); the loop reads the window back after drawing
  // and reports the result.
  std::filesystem::path take_screenshot_request();
  // Scripted screenshots: the help overlay, a pinned hovered point, an in-app screenshot.
  void show_help(bool on) { show_help_ = on; }
  // Turns the main view about the world up by d_yaw radians (scripted orbits, --orbit).
  void orbit(float d_yaw) { camera_.rotate(d_yaw, 0.0F); }
  // Scales the view distance once the view has been fitted (< 1: closer; --zoom).
  void zoom_after_fit(float factor) { zoom_after_fit_ = factor; }
  void set_point_size(float px) { point_size_ = px; }
  // Frustums coloured by physical camera (rig datasets) or by registration order.
  void set_color_by_camera(bool on) {
    color_by_camera_ = on;
    dirty_ = true;
  }
  void set_camera_size(float s) {
    options_.frustum_size = s;
    dirty_ = true;
  }
  void pin_hover(int point) { pinned_hover_ = point; }
  void screenshot_now() { request_screenshot(); }
  void screenshot_done(const std::filesystem::path& file, const std::string& error);
  // Selects image index `image` (shows the Camera tab); for scripted screenshots.
  void select(int image) { select_camera(image); }
  // One UI + render pass (between ImGui::NewFrame and ImGui::Render).
  void frame();
  const std::string& error() const { return error_; }
  bool running() const { return controller_.active(); }

 private:
  enum class Tab : std::uint8_t { kRun, kCamera, kConfig };
  enum class CompareLayout : std::uint8_t { kSideBySide, kOverlay, kWipe };
  struct CompareSide {
    std::filesystem::path dir;
    LoadedRun run;
    pipeline::SceneSnapshot snapshot;  // B: mapped into A's frame
    GroundTruthView gt;
    pipeline::Metrics metrics;
    double seconds = 0.0;
    SceneGeometry geometry;
    std::unique_ptr<gl::Renderer> renderer;
  };
  struct Compare {
    CompareSide a;
    CompareSide b;
    SnapshotAlignment alignment;
    std::vector<ConfigDifference> diff;
    std::vector<MetricRow> rows;
    CompareLayout layout = CompareLayout::kSideBySide;
    bool link = true;
    OrbitCamera camera_b;  // B's own view when not linked
    float wipe = 0.5F;     // divider position (fraction of the view width)
    bool dragging_wipe = false;
    int selected_id = -1;  // dataset id selected in both
    bool dirty = true;
  };
  // Per-stage seconds for the timeline: load, features, matching, tracks, reconstruction without
  // BA, BA, export.
  using StageTimes = std::array<double, 7>;

  void poll_run();
  void reset_scene();
  void header(const Rect& r);
  void left_rail(const Rect& r);
  void inspect_rail(float width);
  void setup_rail(float width);
  void inspector(const Rect& r);
  void run_tab(float width);
  void camera_tab(float width);
  void config_tab();
  void timeline(const Rect& r);
  void viewport_overlays(const Rect& r);
  void help_overlay(const Rect& r);
  void request_screenshot();
  void handle_input(const Rect& view, float fb_scale_x, float fb_scale_y);
  void rebuild();
  void select_camera(int image);
  void look_through_selected();
  void fit_view();
  const pipeline::SceneSnapshot* current() const;
  // Compare mode (compare_view.cpp).
  bool load_compare_side(CompareSide& side, const std::filesystem::path& dir,
                         const SnapshotAlignment* to_a);
  void compare_frame(int fb_width, int fb_height);
  void compare_header(const Rect& r);
  void compare_panel(const Rect& r);
  void compare_chart(const Rect& r);
  void compare_labels(const Rect& view);
  void compare_rebuild();
  void compare_input(const Rect& view, float sx, float sy);
  std::size_t latest() const { return history_.empty() ? 0 : history_.size() - 1; }

  GLFWwindow* window_;
  std::unique_ptr<gl::Renderer> renderer_;
  std::unique_ptr<gl::Texture> thumb_;
  std::unique_ptr<gl::PointColorsCuda> gpu_colors_;  // V5: null without CUDA-GL (CPU colours)
  bool recolor_ = false;                             // only the colour mode changed
  int hovered_point_ = -1;                           // index into the current snapshot's points
  int pinned_hover_ = -1;                            // --hover: shown instead of the mouse hover
  bool hover_stale_ = true;
  bool show_help_ = false;
  std::filesystem::path screenshot_request_;
  std::string status_;  // a short message over the view ("saved …")
  double status_until_ = 0.0;
  int thumb_image_ = -1;  // image index the thumbnail shows
  OrbitCamera camera_;
  SceneOptions options_;
  SceneGeometry geometry_;
  Eigen::Matrix4f view_proj_ = Eigen::Matrix4f::Identity();
  Rect view_rect_;

  // What is shown: a loaded run or a live / finished run of `config_`.
  RunController controller_;
  SnapshotHistory history_;
  std::vector<EventKind> snapshot_kind_;  // per snapshot: camera or BA
  std::vector<RunEvent> log_;             // all events of the current run
  std::optional<pipeline::Metrics> metrics_;
  StageTimes stages_{};
  double total_seconds_ = 0.0;
  config::Config config_;
  bool have_config_ = false;
  std::string config_path_;
  std::string source_;  // "outputs / bunny-sift-tuned" or the config file name
  std::vector<eval::Pose> truth_;
  CameraGroups groups_;  // physical camera per image (rig datasets)
  bool color_by_camera_ = true;
  float zoom_after_fit_ = 1.0F;  // frustums by physical camera when groups_ has them
  GroundTruthView gt_;
  RunState last_state_ = RunState::kIdle;
  std::string run_message_;  // result of the last run ("finished", "cancelled", the error)

  // UI state
  pipeline::Backend detected_;
  BackendChoice backend_ = BackendChoice::kAuto;
  bool setup_mode_ = false;  // left rail shows the run setup instead of the layers
  Tab tab_ = Tab::kRun;
  std::size_t replay_ = 0;
  bool follow_ = true;    // during a run, show the newest snapshot
  bool auto_fit_ = true;  // during a run, keep the view fitted until the user moves it
  bool playing_ = false;
  std::filesystem::path shown_dir_;
  std::filesystem::path last_screenshot_;  // requested, maybe not written yet  // the run directory
                                           // whose reconstruction the view shows
  double play_clock_ = 0.0;
  bool show_points_ = true;
  bool show_gt_ = false;
  bool show_grid_ = true;
  float point_size_ = 3.0F;
  bool dirty_ = true;
  bool fitted_ = false;
  std::unique_ptr<Compare> cmp_;
  std::filesystem::path run_dir_;  // the finished run on disk being shown (load or a live run)
  std::string compare_input_;
  std::string path_input_;
  std::string config_input_;
  std::string error_;
};

// Writes RGB pixels (rows top-down) as a PNG. Throws IoError.
void save_png(const std::filesystem::path& file, int width, int height,
              const std::vector<std::uint8_t>& rgb);

}  // namespace sfm::viewer
