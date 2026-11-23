#include "app.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <stdexcept>

#include <GLFW/glfw3.h>
#include <nlohmann/json.hpp>

#include "sfm/viewer/colormap.hpp"
#include "sfm/viewer/picking.hpp"
#include "sfm/viewer/run_loader.hpp"

#include "imgui.h"
#include "theme.hpp"
#include "widgets.hpp"

namespace sfm::viewer {

namespace {

using ui::begin_overlay;
using ui::begin_panel;
using ui::ButtonKind;
using ui::center_y;
using ui::end_overlay;
using ui::format;
using ui::Icon;
using ui::median;
using ui::ordinal;
using ui::thousands;
using ui::Tone;

const theme::Fonts& font_set() { return theme::fonts(); }

// Index in App::StageTimes of a pipeline stage (load and BA are not pipeline stages).
int stage_slot(pipeline::Stage s) {
  switch (s) {
    case pipeline::Stage::kFeatures:
      return 1;
    case pipeline::Stage::kMatching:
      return 2;
    case pipeline::Stage::kTracks:
      return 3;
    case pipeline::Stage::kReconstruction:
      return 4;
    case pipeline::Stage::kExport:
      return 6;
  }
  return 0;
}

const char* const kStageNames[] = {"load",           "features",          "matching", "tracks",
                                   "reconstruction", "bundle adjustment", "export"};
const ImU32 kStageColors[] = {theme::kTrack,     theme::kAccentStrong, theme::kAccent, theme::kText,
                              theme::kTextFaint, theme::kWarn,         theme::kTrack};

}  // namespace

App::App(GLFWwindow* window)
    : window_(window),
      renderer_(std::make_unique<gl::Renderer>()),
      thumb_(std::make_unique<gl::Texture>()),
      detected_(pipeline::detect_backend()) {
  const char* force_cpu = std::getenv("SFM_VIEWER_CPU_COLORS");
  if ((force_cpu == nullptr || force_cpu[0] == '0') && gl::PointColorsCuda::available()) {
    gpu_colors_ = std::make_unique<gl::PointColorsCuda>();
  }
  path_input_ = "outputs/bunny-gpu";
  config_input_ = "configs/bunny_sift.json";
}

App::~App() = default;

void App::reset_scene() {
  history_.clear();
  snapshot_kind_.clear();
  log_.clear();
  metrics_.reset();
  stages_.fill(0.0);
  total_seconds_ = 0.0;
  gt_ = GroundTruthView{};
  replay_ = 0;
  playing_ = false;
  options_.selected_camera = -1;
  thumb_->reset();
  thumb_image_ = -1;
  fitted_ = false;
  dirty_ = true;
}

bool App::load(const std::filesystem::path& run_dir) {
  if (controller_.active()) {
    error_ = "a run is in progress: cancel it first";
    return false;
  }
  try {
    LoadedRun r = load_run(run_dir);
    reset_scene();
    if (r.replay.empty()) {
      history_.push(r.snapshot);
      snapshot_kind_.push_back(EventKind::kCamera);
    } else {  // ▶ replays the registration order (final poses); opens on the final state
      for (auto& s : r.replay) {
        history_.push(std::move(s));
        snapshot_kind_.push_back(EventKind::kCamera);
      }
    }
    replay_ = latest();
    follow_ = true;
    truth_ = r.truth;
    groups_ = r.groups;
    config_ = r.config;
    have_config_ = true;
    config_path_ = (run_dir / "config.json").string();
    if (!r.metrics_json.empty()) metrics_ = parse_metrics(r.metrics_json);
    if (!r.timings_json.empty()) {
      const auto t = nlohmann::json::parse(r.timings_json);
      const double ba = t.value("ba", 0.0);
      stages_ = {t.value("load", 0.0),
                 t.value("features", 0.0),
                 t.value("matching", 0.0),
                 t.value("tracks", 0.0),
                 std::max(t.value("reconstruction", 0.0) - ba, 0.0),
                 ba,
                 t.value("export", 0.0)};
      total_seconds_ = t.value("total", 0.0);
    }
    const std::filesystem::path abs = std::filesystem::absolute(run_dir).lexically_normal();
    const std::filesystem::path dir = abs.has_filename() ? abs : abs.parent_path();
    source_ = dir.parent_path().filename().string() + " / " + dir.filename().string();
    path_input_ = run_dir.string();
    run_dir_ = run_dir;
    shown_dir_ = run_dir;
    config_input_ = config_path_;  // New run starts from this run's config unless changed
    last_state_ = RunState::kIdle;
    run_message_.clear();
    setup_mode_ = false;
    error_.clear();
    return true;
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
}

bool App::open_config(const std::filesystem::path& file) {
  if (controller_.active()) return false;
  try {
    config_ = config::load_config(file);
    have_config_ = true;
    config_path_ = file.string();
    config_input_ = file.string();
    source_ = "configs / " + file.filename().string();
    try {
      truth_ = load_truth(config_.dataset);
    } catch (const std::exception&) {
      truth_.clear();  // no calibration: no ground truth
    }
    groups_ = load_camera_groups(config_.dataset);
    error_.clear();
    return true;
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
}

bool App::start_run() {
  if (!have_config_ || controller_.active()) return false;
  const config::Config cfg = with_backend(config_, backend_, detected_.gpu);
  if (!controller_.start(pipeline_runner(cfg))) return false;
  reset_scene();
  config_ = cfg;
  follow_ = true;
  auto_fit_ = true;
  run_message_.clear();
  last_state_ = RunState::kRunning;
  setup_mode_ = true;  // the backends stay visible (locked) while it runs
  tab_ = Tab::kRun;
  return true;
}

void App::poll_run() {
  RunUpdate u = controller_.poll();
  if (u.state == RunState::kIdle) return;
  const bool had = !history_.empty();
  for (auto& s : u.snapshots) {
    try {
      history_.push(std::move(s));
    } catch (const std::invalid_argument& e) {
      error_ = e.what();
    }
  }
  for (RunEvent& e : u.events) {
    if (e.snapshot >= 0) snapshot_kind_.push_back(e.kind);
    if (e.kind == EventKind::kStageEnd) {
      stages_[static_cast<std::size_t>(stage_slot(e.stage))] = e.seconds;
    }
    log_.push_back(std::move(e));
  }
  if (u.metrics) metrics_ = u.metrics;
  if (u.timings) {
    const pipeline::Timings& t = *u.timings;
    stages_ = {t.load, t.features, t.matching, t.tracks, std::max(t.reconstruction - t.ba, 0.0),
               t.ba,   t.export_};
    total_seconds_ = t.total;
  }
  if (!u.snapshots.empty()) {
    if (follow_) replay_ = latest();
    if (!had) fitted_ = false;
    dirty_ = true;
  }
  if (u.state != last_state_) {
    last_state_ = u.state;
    if (u.state == RunState::kFinished) {
      run_message_ = "finished";
      setup_mode_ = false;
      run_dir_ = config_.run.output_dir / config_.run.name;
      shown_dir_ = run_dir_;
    } else if (u.state == RunState::kCancelled) {
      run_message_ = "cancelled: the last snapshot is kept";
      if (history_.empty() && !shown_dir_.empty() && load(shown_dir_)) {
        // Cancelled before the reconstruction: show the run that was open before.
        setup_mode_ = true;
        last_state_ = RunState::kCancelled;  // load() reset it; do not handle the cancel twice
        run_message_ = "cancelled early: previous run shown";
      }
    } else if (u.state == RunState::kFailed) {
      run_message_ = log_.empty() ? "failed" : "failed: " + log_.back().message;
    }
    if (follow_) replay_ = latest();
    dirty_ = true;
  }
  if (controller_.active()) total_seconds_ = controller_.elapsed_seconds();
}

const pipeline::SceneSnapshot* App::current() const {
  return history_.empty() ? nullptr : &history_.at(replay_);
}

void App::rebuild() {
  const pipeline::SceneSnapshot* s = current();
  if (s == nullptr) {
    geometry_ = SceneGeometry{};
  } else {
    gt_ = truth_.size() == s->ids.size() ? align_ground_truth(*s, truth_) : GroundTruthView{};
    options_.ground_truth = show_gt_ && gt_.valid ? gt_.aligned : std::vector<eval::Pose>{};
    options_.camera_group = color_by_camera_ && groups_.group.size() == s->ids.size()
                                ? groups_.group
                                : std::vector<int>{};
    geometry_ = build_scene(*s, options_);
  }
  renderer_->set_points(show_points_ ? geometry_.points : std::vector<Vertex>{});
  if (gpu_colors_ && s != nullptr) {
    try {  // the buffer was re-specified: register it again, with this snapshot's attributes
      gpu_colors_->attach(renderer_->points_vbo(), renderer_->point_count());
      if (renderer_->point_count() > 0) {
        gpu_colors_->set_attributes(s->point_gray, s->point_error_px, s->point_track_length);
      }
    } catch (const std::exception& e) {
      gpu_colors_.reset();  // keep working on the CPU path
      error_ = std::string("GPU colours off: ") + e.what();
    }
  }
  recolor_ = false;
  std::vector<Vertex> lines = geometry_.lines;
  if (show_grid_ && s != nullptr) {
    const auto grid = grid_lines(geometry_.center, geometry_.radius, geometry_.up);
    lines.insert(lines.end(), grid.begin(), grid.end());
  }
  renderer_->set_lines(lines);
  hover_stale_ = true;
  if ((!fitted_ || (auto_fit_ && controller_.active())) && s != nullptr) fit_view();
  dirty_ = false;
}

void App::fit_view() {
  camera_.set_up(geometry_.up);
  const float r =
      options_.frustums ? std::max(geometry_.radius, geometry_.camera_radius) : geometry_.radius;
  camera_.fit(geometry_.center, r);
  if (!fitted_ && zoom_after_fit_ != 1.0F) camera_.zoom(zoom_after_fit_);
  fitted_ = true;
}

void App::select_camera(int image) {
  if (image == options_.selected_camera) return;
  options_.selected_camera = image;
  if (image >= 0) tab_ = Tab::kCamera;
  dirty_ = true;
}

void App::look_through_selected() {
  const int i = options_.selected_camera;
  if (i < 0 || static_cast<std::size_t>(i) >= geometry_.camera_centres.size()) {
    status_ = "select a camera first: click one in the view";
    status_until_ = ImGui::GetTime() + 3.0;
    return;
  }
  const Eigen::Vector3f c = geometry_.camera_centres[static_cast<std::size_t>(i)];
  if (!c.allFinite()) return;
  camera_.set_target(geometry_.center);
  camera_.look_from(c);
  auto_fit_ = false;
}

void App::handle_input(const Rect& view, float sx, float sy) {
  ImGuiIO& io = ImGui::GetIO();
  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_F)) fit_view();
    if (ImGui::IsKeyPressed(ImGuiKey_C)) look_through_selected();
    if (ImGui::IsKeyPressed(ImGuiKey_Space) && history_.size() > 1) {
      playing_ = !playing_;
      if (playing_ && replay_ == latest()) replay_ = 0;
      follow_ = false;
      dirty_ = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && controller_.active()) controller_.cancel();
    if (ImGui::IsKeyPressed(ImGuiKey_S)) request_screenshot();
    if (ImGui::IsKeyPressed(ImGuiKey_Slash) && io.KeyShift) show_help_ = !show_help_;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && show_help_) show_help_ = false;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && replay_ > 0) {
      --replay_, follow_ = false, dirty_ = true;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) && replay_ < latest()) ++replay_, dirty_ = true;
  }
  if (io.WantCaptureMouse && pinned_hover_ < 0) {
    hovered_point_ = -1;
    return;
  }
  if (pinned_hover_ >= 0) {
    hovered_point_ = pinned_hover_;
    return;
  }
  // Hover: the front-most point under the cursor, recomputed when the mouse or the view moves.
  if (show_points_ && view.contains(io.MousePos.x, io.MousePos.y) && !ImGui::IsAnyMouseDown()) {
    const bool moved = io.MouseDelta.x != 0.0F || io.MouseDelta.y != 0.0F;
    if (moved || hover_stale_) {
      const int w = static_cast<int>(view.w * sx);
      const int h = static_cast<int>(view.h * sy);
      hovered_point_ =
          pick_point(geometry_.points, view_proj_, w, h,
                     Eigen::Vector2f((io.MousePos.x - view.x) * sx, (io.MousePos.y - view.y) * sy));
      hover_stale_ = false;
    }
  } else {
    hovered_point_ = -1;
    hover_stale_ = true;
  }
  const ImVec2 click = io.MouseClickedPos[0];
  const bool from_view = view.contains(click.x, click.y);
  const ImVec2 d = io.MouseDelta;
  if (from_view && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    camera_.rotate(-d.x * 0.006F, d.y * 0.006F);
    auto_fit_ = false;
  }
  const ImVec2 rclick = io.MouseClickedPos[1];
  if (view.contains(rclick.x, rclick.y) && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    camera_.pan(-d.x / view.h, d.y / view.h);
    auto_fit_ = false;
  }
  const ImVec2 m = io.MousePos;
  if (!view.contains(m.x, m.y)) return;
  if (io.MouseWheel != 0.0F) {
    camera_.zoom(std::pow(0.88F, io.MouseWheel));
    auto_fit_ = false;
  }
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && from_view &&
      ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0F).x == 0.0F &&
      ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0F).y == 0.0F) {
    const int w = static_cast<int>(view.w * sx);
    const int h = static_cast<int>(view.h * sy);
    const Eigen::Vector2f px((m.x - view.x) * sx, (m.y - view.y) * sy);
    select_camera(pick_camera(geometry_.camera_centres, view_proj_, w, h, px));
  }
}

// ---------------------------------------------------------------------------------------------
// Header

void App::header(const Rect& r) {
  if (begin_panel("##header", r, theme::kSurface, ImVec2(20, 0))) {
    ui::panel_frame("b");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    // Brand: logo + "SFM Viewer" (Gelasio).
    ui::draw_icon(dl, Icon::kLogo, ImVec2(o.x + 20, o.y + 15), ImVec2(o.x + 42, o.y + 37),
                  theme::kAccent);
    ImGui::SetCursorPos(ImVec2(52, 0));
    center_y(0, r.h, theme::kTitle + 4);
    ui::text(font_set().display, theme::kTitle, theme::kText, "SFM Viewer");
    // Source: "outputs / run" with the picker popup.
    ImGui::SameLine(0, 18);
    center_y(0, r.h, 30);
    const std::string src = source_.empty() ? "open a run or a config" : source_;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 7));
    ImGui::PushFont(font_set().mono_medium, theme::kBody);
    if (ImGui::Button((src + "     ##source").c_str())) ImGui::OpenPopup("source");
    ImGui::PopFont();
    ImGui::PopStyleVar();
    ui::draw_icon(
        dl, Icon::kChevron, ImVec2(ImGui::GetItemRectMax().x - 24, ImGui::GetItemRectMin().y + 9),
        ImVec2(ImGui::GetItemRectMax().x - 10, ImGui::GetItemRectMin().y + 23), theme::kTextMuted);
    if (ImGui::BeginPopup("source")) {
      if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
      ui::eyebrow("OPEN A FINISHED RUN");
      char buf[512];
      std::snprintf(buf, sizeof buf, "%s", path_input_.c_str());
      ImGui::SetNextItemWidth(360);
      if (ImGui::InputText("##run", buf, sizeof buf)) path_input_ = buf;
      ImGui::SameLine();
      if (ui::button("Load", ButtonKind::kPrimary, ImVec2(0, 30), !controller_.active())) {
        if (load(path_input_)) ImGui::CloseCurrentPopup();
      }
      ImGui::Spacing();
      ui::eyebrow("OR A CONFIG TO RUN");
      std::snprintf(buf, sizeof buf, "%s", config_input_.c_str());
      ImGui::SetNextItemWidth(360);
      if (ImGui::InputText("##cfg", buf, sizeof buf)) config_input_ = buf;
      ImGui::SameLine();
      if (ui::button("Open", ButtonKind::kSecondary, ImVec2(0, 30), !controller_.active())) {
        if (open_config(config_input_)) {
          setup_mode_ = true;
          ImGui::CloseCurrentPopup();
        }
      }
      if (!error_.empty()) ui::text(font_set().ui, theme::kSmall, theme::kWarn, error_.c_str());
      ImGui::EndPopup();
    }
    // Method chips (or the live status while running).
    if (controller_.active()) {
      ImGui::SameLine(0, 16);
      const pipeline::SceneSnapshot* s = history_.empty() ? nullptr : &history_.at(latest());
      std::string stage = "Starting";
      for (auto it = log_.rbegin(); it != log_.rend(); ++it) {
        if (it->kind == EventKind::kStageBegin) {
          stage = pipeline::to_string(it->stage);
          stage[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(stage[0])));
          break;
        }
      }
      if (stage == "Reconstruction" && s != nullptr) {
        stage = format("Reconstructing · %zu / %zu", s->order.size(), s->ids.size());
      }
      const ImVec2 ts = ui::text_size(font_set().ui, theme::kSmall, stage.c_str());
      const ImVec2 p(ImGui::GetCursorScreenPos().x, o.y + (r.h - 26) * 0.5F);
      const ImVec2 q(p.x + ts.x + 34, p.y + 26);
      dl->AddRectFilled(p, q, theme::kRaised, 13);
      dl->AddRect(p, q, theme::kGood, 13);
      dl->AddCircleFilled(ImVec2(p.x + 14, p.y + 13), 3.5F, theme::kGood);
      dl->AddText(font_set().ui, theme::kSmall, ImVec2(p.x + 24, p.y + (26 - ts.y) * 0.5F),
                  theme::kGood, stage.c_str());
      const std::string el = format("%.1f s", total_seconds_);
      const ImVec2 es = ui::text_size(font_set().mono, theme::kSmall, el.c_str());
      dl->AddText(font_set().mono, theme::kSmall,
                  ImVec2(q.x + 12, o.y + std::floor((r.h - es.y) * 0.5F)), theme::kTextMuted,
                  el.c_str());
      ImGui::Dummy(ImVec2(q.x - p.x + 12 + es.x, 1));
    } else if (have_config_) {
      for (const std::string* m :
           {&config_.features.method, &config_.matching.method, &config_.ba.method}) {
        ImGui::SameLine(0, 6);
        const ImVec2 ts = ui::text_size(font_set().mono, theme::kSmall, m->c_str());
        const ImVec2 p(ImGui::GetCursorScreenPos().x, o.y + (r.h - 22) * 0.5F);
        dl->AddRect(p, ImVec2(p.x + ts.x + 16, p.y + 22), theme::kLine, 11);
        dl->AddText(font_set().mono, theme::kSmall, ImVec2(p.x + 8, p.y + (22 - ts.y) * 0.5F),
                    theme::kTextMuted, m->c_str());
        ImGui::Dummy(ImVec2(ts.x + 16, 1));
      }
    }
    // Right side.
    float x = r.w - 16;
    auto place = [&](float w) {
      x -= w;
      ImGui::SameLine(x);
    };
    if (controller_.active()) {
      const float bw = ui::text_size(font_set().ui, theme::kBody, "Cancel run").x + 28;
      place(bw);
      center_y(0, r.h, 32);
      if (ui::button("Cancel run", ButtonKind::kDanger)) controller_.cancel();
      place(8 + ui::text_size(font_set().mono, theme::kCaption, "Esc").x + 10);
      center_y(0, r.h, 18);
      ui::keycap("Esc");
      place(18 + 24 + ui::text_size(font_set().ui, theme::kBody, "Follow newest").x);
      center_y(0, r.h, 18);
      if (ui::checkbox("Follow newest", follow_) && follow_) replay_ = latest(), dirty_ = true;
    } else {
      const float nw = ui::text_size(font_set().ui, theme::kBody, "New run").x + 28;
      place(nw);
      center_y(0, r.h, 32);
      if (ui::button("New run", ButtonKind::kPrimary)) {
        setup_mode_ = true;
        if (!have_config_) open_config(config_input_);
      }
      const float cw = ui::text_size(font_set().ui, theme::kBody, "Compare").x + 28;
      place(cw + 8);
      center_y(0, r.h, 32);
      if (ui::button("Compare", ButtonKind::kGhost, ImVec2(0, 32), !run_dir_.empty())) {
        ImGui::OpenPopup("compare_pick");
      }
      if (run_dir_.empty() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("open a finished run first: it becomes A");
      }
      if (ImGui::BeginPopup("compare_pick")) {
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
        ui::eyebrow("COMPARE WITH RUN B (A FINISHED RUN DIRECTORY)");
        char buf[512];
        std::snprintf(buf, sizeof buf, "%s", compare_input_.c_str());
        ImGui::SetNextItemWidth(360);
        if (ImGui::InputText("##cmp_b", buf, sizeof buf)) compare_input_ = buf;
        ImGui::SameLine();
        if (ui::button("Compare", ButtonKind::kPrimary, ImVec2(0, 30))) {
          if (open_compare(compare_input_)) ImGui::CloseCurrentPopup();
        }
        if (!error_.empty()) ui::text(font_set().ui, theme::kSmall, theme::kWarn, error_.c_str());
        ImGui::EndPopup();
      }
      const std::string gpu =
          detected_.gpu ? detected_.device : std::string("No CUDA device · CPU");
      const float gw = ui::text_size(font_set().ui, theme::kSmall, gpu.c_str()).x + 14;
      place(gw + 16);
      const ImVec2 ts = ui::text_size(font_set().ui, theme::kSmall, gpu.c_str());
      const ImVec2 p(ImGui::GetCursorScreenPos().x, o.y + std::floor((r.h - ts.y) * 0.5F));
      dl->AddCircleFilled(ImVec2(p.x + 3, p.y + ts.y * 0.5F), 3,
                          detected_.gpu ? theme::kGood : theme::kTextFaint);
      dl->AddText(font_set().ui, theme::kSmall, ImVec2(p.x + 12, p.y), theme::kTextMuted,
                  gpu.c_str());
      ImGui::Dummy(ImVec2(gw, 1));
    }
  }
  ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Left rail

void App::left_rail(const Rect& r) {
  if (begin_panel("##left", r, theme::kSurface)) {
    ui::panel_frame("r");
    const float w = r.w - 32;
    if (setup_mode_ || controller_.active()) {
      setup_rail(w);
    } else {
      inspect_rail(w);
    }
  }
  ImGui::End();
}

void App::inspect_rail(float w) {
  const pipeline::SceneSnapshot* s = current();
  ui::eyebrow("LAYERS");
  ImGui::Dummy(ImVec2(1, 2));
  const std::string npts = s != nullptr ? thousands(static_cast<long long>(s->points.size())) : "";
  const std::string ncam = s != nullptr ? std::to_string(s->order.size()) : "";
  if (ui::layer_row("##points", theme::kText, "Points", npts.c_str(), show_points_, w, true)) {
    dirty_ = true;
  }
  if (ui::layer_row("##cameras", theme::kAccent, "Cameras", ncam.c_str(), options_.frustums, w)) {
    dirty_ = true;
  }
  bool gt = show_gt_;
  if (ui::layer_row("##gt", theme::kWarn, "Ground truth", gt_.valid ? "" : "n/a", gt, w) &&
      gt_.valid) {
    show_gt_ = gt;
    dirty_ = true;
  }
  if (ui::layer_row("##grid", theme::kTextDim, "Grid", "", show_grid_, w)) dirty_ = true;

  ImGui::Dummy(ImVec2(1, 14));
  ui::eyebrow("COLOUR BY");
  int mode = static_cast<int>(options_.color);
  if (ui::segmented("colour", {"Gray", "Height", "Reprojection", "Track length"}, mode, 2, w)) {
    options_.color = static_cast<ColorMode>(mode);
    recolor_ = true;  // GPU: only the colours change (V5); CPU: falls back to a rebuild
  }
  // Legend card.
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(w, 68));
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRect(p, ImVec2(p.x + w, p.y + 68), theme::kLineSoft, 8);
  ImGui::SetCursorScreenPos(ImVec2(p.x + 10, p.y + 9));
  const bool gray = options_.color == ColorMode::kGray;
  const char* names[] = {"", "Points · height", "Points · reprojection error (px)",
                         "Points · track length"};
  ui::text(font_set().ui, theme::kCaption, theme::kTextMuted,
           gray ? (!options_.camera_group.empty() ? "Cameras · rig camera"
                                                  : "Cameras · registration order")
                : names[mode]);
  {  // where the point colours are computed (V5)
    const char* where = gpu_colors_ ? "GPU" : "CPU";
    const ImVec2 ts = ui::text_size(font_set().mono, theme::kCaption, where);
    ImGui::GetWindowDrawList()->AddText(font_set().mono, theme::kCaption,
                                        ImVec2(p.x + w - 10 - ts.x, p.y + 9), theme::kTextFaint,
                                        where);
  }
  const bool by_camera = gray && !options_.camera_group.empty();
  if (by_camera) {  // one swatch per physical camera instead of the order gradient
    const auto k = static_cast<float>(groups_.labels.size());
    const float cw = (w - 20) / std::max(1.0F, k);
    for (std::size_t g = 0; g < groups_.labels.size(); ++g) {
      const Rgb c = camera_group_color(static_cast<int>(g));
      const float x = p.x + 10 + cw * static_cast<float>(g);
      dl->AddRectFilled(ImVec2(x, p.y + 30), ImVec2(x + 14, p.y + 40),
                        IM_COL32(c[0], c[1], c[2], 255), 2);
      dl->AddText(font_set().mono, theme::kCaption, ImVec2(x + 20, p.y + 27), theme::kTextMuted,
                  groups_.labels[g].c_str());
    }
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + 68 + 8));
  }
  if (!by_camera) {
    ImGui::SetCursorScreenPos(ImVec2(p.x + 10, p.y + 28));
    ui::gradient_bar(ImVec2(w - 20, 10));
    ImGui::SetCursorScreenPos(ImVec2(p.x + 10, p.y + 44));
    const int n = s != nullptr ? static_cast<int>(s->order.size()) : 0;
    const std::string lo = gray ? "1st" : format("%.3g", geometry_.color_range[0]);
    const std::string hi =
        gray ? (n > 0 ? ordinal(n) : "last") : format("%.3g", geometry_.color_range[1]);
    ui::text(font_set().mono, theme::kCaption, theme::kTextMuted, lo.c_str());
    ui::text_right(font_set().mono, theme::kCaption, theme::kTextMuted, hi.c_str(),
                   p.x - ImGui::GetWindowPos().x + w - 10);
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + 68 + 8));
  }

  ImGui::Dummy(ImVec2(1, 14));
  ui::eyebrow("DISPLAY");
  ImGui::Dummy(ImVec2(1, 2));
  ui::slider_row("Point size", format("%.1f px", point_size_).c_str(), point_size_, 1.0F, 10.0F, w);
  ImGui::Dummy(ImVec2(1, 6));
  if (ui::slider_row("Camera size", format("%.2f", options_.frustum_size).c_str(),
                     options_.frustum_size, 0.02F, 0.5F, w)) {
    dirty_ = true;
  }
  if (!groups_.labels.empty()) {  // rig datasets: frustums by physical camera or by order
    ImGui::Dummy(ImVec2(1, 6));
    int by = color_by_camera_ ? 1 : 0;
    if (ui::segmented("camera colour", {"Order", "Rig camera"}, by, 2, w)) {
      color_by_camera_ = by == 1;
      dirty_ = true;
    }
  }

  // Shortcuts at the bottom.
  const float bottom = ImGui::GetWindowHeight() - 20 - 3 * 24;
  if (ImGui::GetCursorPosY() < bottom) ImGui::SetCursorPosY(bottom);
  for (const auto& [k, d] :
       {std::pair{"F", "Fit view"}, std::pair{"C", "Look through selected camera"},
        std::pair{"?", "All shortcuts"}}) {
    ui::keycap(k);
    ImGui::SameLine(0, 10);
    ui::text(font_set().ui, theme::kSmall, theme::kTextMuted, d);
  }
}

void App::setup_rail(float w) {
  const float x0 = ImGui::GetCursorPosX();
  const bool locked = controller_.active();
  ui::eyebrow(locked ? "BACKENDS" : "RUN SETUP");
  ImGui::Dummy(ImVec2(1, 4));
  if (!locked) {
    ui::text(font_set().ui, theme::kBody, theme::kText, "Config");
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s", config_input_.c_str());
    ImGui::SetNextItemWidth(w - 62);
    ImGui::PushFont(font_set().mono, theme::kSmall);
    if (ImGui::InputText("##config", buf, sizeof buf)) config_input_ = buf;
    ImGui::PopFont();
    ImGui::SameLine(0, 6);
    if (ui::button("Open", ButtonKind::kSecondary, ImVec2(56, 30))) open_config(config_input_);
    if (have_config_) {
      ui::text(font_set().mono, theme::kCaption, theme::kTextFaint,
               (config_.dataset.image_count() < 0
                    ? format("%s · all images", config_.dataset.root.filename().string().c_str())
                    : format("%s · %d images", config_.dataset.root.filename().string().c_str(),
                             config_.dataset.image_count()))
                   .c_str());
    }
    ImGui::Dummy(ImVec2(1, 10));
  }
  if (locked) ImGui::BeginDisabled();
  int b = static_cast<int>(backend_);
  ui::text(font_set().ui, theme::kBody, theme::kText, "Backend");
  if (ui::segmented("backend", {"Auto", "GPU", "CPU"}, b, 3, w, detected_.gpu ? -1 : 1)) {
    backend_ = static_cast<BackendChoice>(b);
  }
  if (locked) ImGui::EndDisabled();
  ImGui::Dummy(ImVec2(1, 4));
  if (have_config_) {
    const config::Config c = locked ? config_ : with_backend(config_, backend_, detected_.gpu);
    for (const auto& [stage, m] :
         {std::pair{"Features", c.features.method}, std::pair{"Matching", c.matching.method},
          std::pair{"Bundle adjustment", c.ba.method}}) {
      ui::text(font_set().ui, theme::kBody, theme::kText, stage);
      ui::text_right(font_set().mono, theme::kSmall, theme::kTextMuted, m.c_str(), x0 + w);
    }
  }
  ImGui::Dummy(ImVec2(1, 8));
  // Note card.
  const char* note =
      locked ? "Locked while running. Cancel stops at the next check and keeps the last snapshot."
             : (detected_.gpu ? "Auto uses the GPU methods; CPU gives the reference results."
                              : "No CUDA device: runs use the CPU methods.");
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::PushTextWrapPos(x0 + w - 8);
  ImGui::SetCursorScreenPos(ImVec2(p.x + 30, p.y + 10));
  ImGui::PushFont(font_set().ui, theme::kSmall);
  ImGui::PushStyleColor(ImGuiCol_Text, theme::kTextMuted);
  ImGui::TextUnformatted(note);
  ImGui::PopStyleColor();
  ImGui::PopFont();
  ImGui::PopTextWrapPos();
  const float h = ImGui::GetCursorScreenPos().y - p.y + 6;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRect(p, ImVec2(p.x + w, p.y + h), theme::kLine, 8);
  ui::draw_icon(dl, Icon::kLock, ImVec2(p.x + 8, p.y + 10), ImVec2(p.x + 24, p.y + 26),
                theme::kTextMuted);
  ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h + 8));
  if (!run_message_.empty() && !locked) {
    ui::text(font_set().ui, theme::kSmall,
             last_state_ == RunState::kFailed ? theme::kWarn : theme::kTextMuted,
             run_message_.c_str());
  }
  if (!error_.empty() && !locked) {
    ImGui::PushTextWrapPos(x0 + w);
    ui::text(font_set().ui, theme::kSmall, theme::kWarn, error_.c_str());
    ImGui::PopTextWrapPos();
  }
  // Bottom buttons.
  const float bottom = ImGui::GetWindowHeight() - 20 - 32 - (locked ? 0.0F : 40.0F);
  if (ImGui::GetCursorPosY() < bottom) ImGui::SetCursorPosY(bottom);
  if (ui::button("Start run", ButtonKind::kPrimary, ImVec2(w, 32), have_config_ && !locked)) {
    // The config in the field runs, also when it was typed without pressing Open.
    if (config_input_ == config_path_ || open_config(config_input_)) start_run();
  }
  if (!locked) {
    if (ui::button("Back to layers", ButtonKind::kGhost, ImVec2(w, 32))) setup_mode_ = false;
  }
}

// ---------------------------------------------------------------------------------------------
// Inspector

void App::inspector(const Rect& r) {
  if (begin_panel("##right", r, theme::kSurface, ImVec2(0, 0))) {
    ui::panel_frame("l");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    // Tabs.
    float x = 16;
    for (const auto& [t, name] : {std::pair{Tab::kRun, "Run"}, std::pair{Tab::kCamera, "Camera"},
                                  std::pair{Tab::kConfig, "Config"}}) {
      const ImVec2 ts = ui::text_size(font_set().ui, theme::kBody, name);
      ImGui::SetCursorPos(ImVec2(x, 0));
      if (ImGui::InvisibleButton(name, ImVec2(ts.x, 44))) tab_ = t;
      const bool sel = tab_ == t;
      dl->AddText(font_set().ui, theme::kBody, ImVec2(o.x + x, o.y + (44 - ts.y) * 0.5F),
                  sel ? theme::kText : theme::kTextMuted, name);
      if (sel) {
        dl->AddRectFilled(ImVec2(o.x + x, o.y + 42), ImVec2(o.x + x + ts.x, o.y + 44),
                          theme::kAccent);
      }
      x += ts.x + 20;
    }
    dl->AddLine(ImVec2(o.x, o.y + 44), ImVec2(o.x + r.w, o.y + 44), theme::kLine);
    ImGui::SetCursorPos(ImVec2(0, 45));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16, 16));
    if (ImGui::BeginChild("##inspector_body", ImVec2(r.w - 1, r.h - 45),
                          ImGuiChildFlags_AlwaysUseWindowPadding)) {
      const float w = r.w - 33;
      switch (tab_) {
        case Tab::kRun:
          run_tab(w);
          break;
        case Tab::kCamera:
          camera_tab(w);
          break;
        case Tab::kConfig:
          config_tab();
          break;
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
  }
  ImGui::End();
}

void App::run_tab(float w) {
  const float x0 = ImGui::GetCursorPosX();  // content left edge: right-aligned text ends at x0 + w
  const pipeline::SceneSnapshot* s = current();
  if (s == nullptr) {
    if (controller_.active()) {
      ui::text(font_set().ui, theme::kBody, theme::kTextMuted,
               "Running: features and matching first.");
      ui::text(font_set().ui, theme::kSmall, theme::kTextFaint,
               "Cameras and points appear when the reconstruction starts.");
    } else {
      ui::text(font_set().ui, theme::kBody, theme::kTextMuted, "No run yet.");
      ui::text(font_set().ui, theme::kSmall, theme::kTextFaint,
               "Open a finished run, or New run to reconstruct.");
    }
    return;
  }
  const bool final_state = metrics_.has_value() && replay_ == latest() && !controller_.active();
  std::vector<double> rot;
  for (double v : gt_.rotation_deg) {
    if (v >= 0) rot.push_back(v);
  }
  const double rot_med = final_state && metrics_->rotation_median_deg >= 0
                             ? metrics_->rotation_median_deg
                             : median(rot);
  const double rot_max = final_state && metrics_->rotation_max_deg >= 0
                             ? metrics_->rotation_max_deg
                             : (rot.empty() ? -1 : *std::max_element(rot.begin(), rot.end()));
  std::vector<double> err(s->point_error_px.begin(), s->point_error_px.end());
  const double rep = final_state ? metrics_->reproj_median_px : median(err);
  const float cw = (w - 8) * 0.5F;
  ui::stat_card("c1", "CAMERAS", format("%zu/%zu", s->order.size(), s->ids.size()).c_str(),
                "registered", Tone::kNeutral, cw);
  ImGui::SameLine(0, 8);
  ui::stat_card(
      "c2", "POINTS", thousands(static_cast<long long>(s->points.size())).c_str(),
      format("%s observations", final_state ? thousands(metrics_->observations).c_str() : "live")
          .c_str(),
      Tone::kNeutral, cw);
  ui::stat_card("c3", "ROTATION", rot_med >= 0 ? format("%.3f°", rot_med).c_str() : "–",
                rot_max >= 0 ? format("median · max %.3f°", rot_max).c_str() : "needs ground truth",
                Tone::kNeutral, cw);
  ImGui::SameLine(0, 8);
  ui::stat_card("c4", "REPROJECTION", rep >= 0 ? format("%.3f px", rep).c_str() : "–", "median",
                Tone::kNeutral, cw);

  ImGui::Dummy(ImVec2(1, 10));
  ui::eyebrow("STAGES");
  ImGui::Dummy(ImVec2(1, 2));
  for (std::size_t i = 0; i < stages_.size(); ++i) {
    if (stages_[i] <= 0) continue;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + 5), ImVec2(p.x + 8, p.y + 13),
                                              kStageColors[i], 2);
    ImGui::SetCursorScreenPos(ImVec2(p.x + 16, p.y));
    ui::text(font_set().ui, theme::kSmall, theme::kText, kStageNames[i]);
    ui::text_right(font_set().mono, theme::kSmall, theme::kTextMuted,
                   format("%.3f s", stages_[i]).c_str(), x0 + w);
  }
  if (total_seconds_ > 0) {
    ui::text(font_set().ui, theme::kSmall, theme::kTextMuted, "total");
    ui::text_right(font_set().mono_medium, theme::kSmall, theme::kText,
                   format("%.2f s", total_seconds_).c_str(), x0 + w);
  }

  ImGui::Dummy(ImVec2(1, 10));
  ui::eyebrow("EVENTS");
  ui::text_right(font_set().ui, theme::kCaption, theme::kTextFaint, "newest first", x0 + w);
  ImGui::Dummy(ImVec2(1, 2));
  int shown = 0;
  int prev_points = 0;
  std::vector<int> points_before(log_.size(), 0);
  for (std::size_t i = 0; i < log_.size(); ++i) {
    points_before[i] = prev_points;
    if (log_[i].snapshot >= 0) prev_points = log_[i].points;
  }
  for (std::size_t k = log_.size(); k-- > 0 && shown < 60;) {
    const RunEvent& e = log_[k];
    std::string msg;
    std::string meta;
    ImU32 dot = theme::kAccent;
    bool round = false;
    switch (e.kind) {
      case EventKind::kCamera:
        msg = e.cameras <= 2 ? "initial pair"
                             : format("registered %04d",
                                      s->ids.empty() || e.newest_image < 0
                                          ? e.newest_image
                                          : s->ids[static_cast<std::size_t>(e.newest_image)]);
        meta = format("+%d", e.points - points_before[k]);
        break;
      case EventKind::kBundleAdjustment:
        msg = format("bundle adjustment · %d cams", e.cameras);
        meta = thousands(e.points);
        dot = theme::kWarn;
        round = true;
        break;
      case EventKind::kStageEnd:
        msg = pipeline::to_string(e.stage) + " done";
        meta = format("%.2f s", e.seconds);
        dot = theme::kTextFaint;
        break;
      case EventKind::kFinished:
        msg = "finished";
        meta = format("%.2f s", e.seconds);
        dot = theme::kGood;
        round = true;
        break;
      case EventKind::kCancelled:
        msg = "cancelled";
        dot = theme::kWarn;
        break;
      case EventKind::kFailed:
        msg = "failed: " + e.message;
        dot = theme::kWarn;
        break;
      case EventKind::kStageBegin:
        continue;
    }
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (shown == 0) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + 26), theme::kRaised, 6);
    if (round) {
      dl->AddCircleFilled(ImVec2(p.x + 12, p.y + 13), 3.5F, dot);
    } else {
      dl->AddRectFilled(ImVec2(p.x + 8.5F, p.y + 9.5F), ImVec2(p.x + 15.5F, p.y + 16.5F), dot, 1);
    }
    ImGui::SetCursorScreenPos(ImVec2(p.x + 26, p.y + 5));
    ui::text(font_set().mono, theme::kSmall, shown == 0 ? theme::kText : theme::kTextMuted,
             msg.c_str());
    ui::text_right(font_set().mono, theme::kSmall, theme::kTextFaint, meta.c_str(), x0 + w - 8);
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + 27));
    ++shown;
  }
  ImGui::Dummy(ImVec2(1, 1));
}

void App::camera_tab(float w) {
  const float x0 = ImGui::GetCursorPosX();
  const pipeline::SceneSnapshot* s = current();
  const int sel = options_.selected_camera;
  if (s == nullptr || sel < 0 || static_cast<std::size_t>(sel) >= s->ids.size()) {
    ui::text(font_set().ui, theme::kBody, theme::kTextMuted, "No camera selected.");
    ui::text(font_set().ui, theme::kSmall, theme::kTextFaint,
             "Click a camera in the view, or a bar below.");
  }
  const auto i = static_cast<std::size_t>(std::max(sel, 0));
  if (s != nullptr && sel >= 0 && i < s->ids.size()) {
    if (thumb_image_ != sel && have_config_) {
      thumb_->load(image_path(config_.dataset, s->ids[i]));
      thumb_image_ = sel;
    }
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float th = w * 0.75F;
    if (thumb_->id() != 0) {
      ImGui::GetWindowDrawList()->AddImageRounded(static_cast<ImTextureID>(thumb_->id()), p,
                                                  ImVec2(p.x + w, p.y + th), ImVec2(0, 0),
                                                  ImVec2(1, 1), IM_COL32_WHITE, 10);
    } else {
      ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + w, p.y + th), theme::kRaised, 10);
    }
    // Worst-rotation badge.
    double worst = -1;
    for (double v : gt_.rotation_deg) worst = std::max(worst, v);
    const bool is_worst = gt_.valid && worst >= 0 && gt_.rotation_deg[i] == worst;
    if (is_worst) {
      const char* b = "Worst rotation";
      const ImVec2 ts = ui::text_size(font_set().ui, theme::kCaption, b);
      ImDrawList* dl = ImGui::GetWindowDrawList();
      dl->AddRectFilled(ImVec2(p.x + 8, p.y + 8), ImVec2(p.x + 24 + ts.x, p.y + 12 + ts.y),
                        theme::kWarn, 10);
      dl->AddText(font_set().ui, theme::kCaption, ImVec2(p.x + 16, p.y + 10), theme::kOnWarn, b);
    }
    ImGui::Dummy(ImVec2(w, th));
    ImGui::Dummy(ImVec2(1, 4));
    ui::text(font_set().display, theme::kHero, theme::kText, format("%04d", s->ids[i]).c_str());
    const auto pos = std::find(s->order.begin(), s->order.end(), sel);
    const std::string reg = pos == s->order.end()
                                ? std::string("not registered")
                                : "registered " +
                                      ordinal(static_cast<int>(pos - s->order.begin()) + 1) +
                                      " of " + std::to_string(s->ids.size());
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
    ui::text_right(font_set().ui, theme::kSmall, theme::kTextMuted, reg.c_str(), x0 + w);
    ImGui::Dummy(ImVec2(1, 2));
    const float cw = (w - 8) * 0.5F;
    std::vector<double> rot;
    for (double v : gt_.rotation_deg) {
      if (v >= 0) rot.push_back(v);
    }
    std::vector<double> cen;
    for (double v : gt_.centre_pct) {
      if (v >= 0) cen.push_back(v);
    }
    const bool has = gt_.valid && gt_.rotation_deg[i] >= 0;
    ui::stat_card("r", "ROTATION", has ? format("%.3f°", gt_.rotation_deg[i]).c_str() : "–",
                  has ? format("run median %.3f°", median(rot)).c_str() : "no ground truth",
                  is_worst ? Tone::kWarn : Tone::kNeutral, cw);
    ImGui::SameLine(0, 8);
    ui::stat_card("t", "CENTRE", has ? format("%.3f %%", gt_.centre_pct[i]).c_str() : "–",
                  has ? format("run median %.3f %%", median(cen)).c_str() : "", Tone::kNeutral, cw);
  }
  // Per-camera rotation error bars (click selects).
  if (s != nullptr && gt_.valid) {
    ImGui::Dummy(ImVec2(1, 8));
    ui::eyebrow(format("ROTATION ERROR · ALL %zu CAMERAS", s->ids.size()).c_str());
    const ImVec2 p = ImGui::GetCursorScreenPos();
    constexpr float kH = 64;
    ImGui::InvisibleButton("##bars", ImVec2(w, kH));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const std::size_t n = gt_.rotation_deg.size();
    double top = 0;
    for (double v : gt_.rotation_deg) top = std::max(top, v);
    top = std::max(top * 1.1, 1e-6);
    const float bw = (w - 2.0F * static_cast<float>(n - 1)) / static_cast<float>(n);
    double worst = -1;
    for (double v : gt_.rotation_deg) worst = std::max(worst, v);
    for (std::size_t k = 0; k < n; ++k) {
      const double v = gt_.rotation_deg[k];
      if (v < 0) continue;
      const float bx = p.x + static_cast<float>(k) * (bw + 2);
      const float h = std::max(2.0F, static_cast<float>(v / top) * kH);
      const ImU32 c = static_cast<int>(k) == options_.selected_camera
                          ? theme::kAccent
                          : (v == worst ? theme::kWarn : theme::kBarNeutral);
      dl->AddRectFilled(ImVec2(bx, p.y + kH - h), ImVec2(bx + bw, p.y + kH), c, 2,
                        ImDrawFlags_RoundCornersTop);
    }
    dl->AddLine(ImVec2(p.x, p.y + kH), ImVec2(p.x + w, p.y + kH), theme::kLine);
    if (ImGui::IsItemClicked()) {
      const int k = static_cast<int>((ImGui::GetIO().MousePos.x - p.x) / (bw + 2));
      if (k >= 0 && static_cast<std::size_t>(k) < n &&
          gt_.rotation_deg[static_cast<std::size_t>(k)] >= 0) {
        select_camera(k);
      }
    }
    if (ImGui::IsItemHovered()) {
      const int k = static_cast<int>((ImGui::GetIO().MousePos.x - p.x) / (bw + 2));
      if (k >= 0 && static_cast<std::size_t>(k) < n &&
          gt_.rotation_deg[static_cast<std::size_t>(k)] >= 0) {
        ImGui::SetTooltip("%04d  %.3f deg", s->ids[static_cast<std::size_t>(k)],
                          gt_.rotation_deg[static_cast<std::size_t>(k)]);
      }
    }
    ui::text(font_set().mono, theme::kCaption, theme::kTextFaint,
             format("%04d", s->ids.front()).c_str());
    ui::text_right(font_set().mono, theme::kCaption, theme::kTextFaint,
                   format("%04d", s->ids.back()).c_str(), x0 + w);
  }
  if (s != nullptr && sel >= 0) {
    ImGui::Dummy(ImVec2(1, 6));
    if (ui::button("Look through camera", ButtonKind::kSecondary, ImVec2(w, 34))) {
      look_through_selected();
    }
  }
}

void App::config_tab() {
  if (!have_config_) {
    ui::text(font_set().ui, theme::kBody, theme::kTextMuted, "No config.");
    return;
  }
  ui::text(font_set().mono, theme::kCaption, theme::kTextFaint, config_path_.c_str());
  ImGui::Dummy(ImVec2(1, 4));
  const std::string json = config::to_json(config_);
  ImGui::PushFont(font_set().mono, theme::kSmall);
  ImGui::PushStyleColor(ImGuiCol_Text, theme::kTextMuted);
  ImGui::TextUnformatted(json.c_str());
  ImGui::PopStyleColor();
  ImGui::PopFont();
}

// ---------------------------------------------------------------------------------------------
// Timeline

void App::timeline(const Rect& r) {
  if (begin_panel("##timeline", r, theme::kSurface, ImVec2(20, 0))) {
    ui::panel_frame("t");
    const ImVec2 o = ImGui::GetWindowPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = r.h * 0.5F;
    // Transport.
    ImGui::SetCursorPos(ImVec2(20, cy - 17));
    const bool many = history_.size() > 1;
    if (!many) ImGui::BeginDisabled();
    const ImVec2 b0 = ImGui::GetCursorScreenPos();
    if (ui::icon_button("##back", Icon::kBack, false, 0, 34) && replay_ > 0) {
      --replay_, follow_ = false, dirty_ = true;
    }
    dl->AddRect(b0, ImVec2(b0.x + 34, b0.y + 34), theme::kLine, 8);
    ImGui::SameLine(0, 4);
    if (ui::icon_button("##play", playing_ ? Icon::kPause : Icon::kPlay, false,
                        many ? theme::kAccent : 0, 34)) {
      playing_ = !playing_;
      if (playing_ && replay_ == latest()) replay_ = 0;
      follow_ = false;
      dirty_ = true;
    }
    ImGui::SameLine(0, 4);
    const ImVec2 b2 = ImGui::GetCursorScreenPos();
    if (ui::icon_button("##fwd", Icon::kForward, false, 0, 34) && replay_ < latest()) {
      ++replay_, dirty_ = true;
    }
    dl->AddRect(b2, ImVec2(b2.x + 34, b2.y + 34), theme::kLine, 8);
    if (!many) ImGui::EndDisabled();

    // Totals on the right.
    const std::string total = total_seconds_ > 0 ? format("%.2f s", total_seconds_) : "–";
    const std::string count =
        format("%zu / %zu events", history_.empty() ? 0 : replay_ + 1, history_.size());
    const float tw =
        std::max(ui::text_size(font_set().mono_medium, theme::kValueL, total.c_str()).x,
                 ui::text_size(font_set().mono, theme::kSmall, count.c_str()).x);
    const float right = r.w - 20;
    dl->AddText(
        font_set().mono_medium, theme::kValueL,
        ImVec2(o.x + right - ui::text_size(font_set().mono_medium, theme::kValueL, total.c_str()).x,
               o.y + cy - 20),
        theme::kText, total.c_str());
    dl->AddText(font_set().mono, theme::kSmall,
                ImVec2(o.x + right - ui::text_size(font_set().mono, theme::kSmall, count.c_str()).x,
                       o.y + cy + 2),
                theme::kTextMuted, count.c_str());

    // Scrubber.
    const float x0 = 20 + 3 * 34 + 8 + 20;
    const float sw = right - tw - 20 - x0;
    const float sy = cy - 26;
    ImGui::SetCursorPos(ImVec2(x0, sy));
    const int n = static_cast<int>(history_.size());
    constexpr float kMark = 9;
    ImGui::InvisibleButton("##scrub", ImVec2(sw, 24));
    if (ImGui::IsItemActive() && n > 0) {
      const int k = nearest_mark(ImGui::GetIO().MousePos.x - (o.x + x0), n, sw, kMark);
      if (static_cast<std::size_t>(k) != replay_) {
        replay_ = static_cast<std::size_t>(k);
        dirty_ = true;
      }
      follow_ = replay_ == latest();
      playing_ = false;
    }
    const float ly = o.y + sy + 12;
    const auto xs = mark_positions(n, sw, kMark);
    const float cur = n > 0 ? xs[replay_] + kMark * 0.5F : 0;
    dl->AddRectFilled(ImVec2(o.x + x0, ly - 1), ImVec2(o.x + x0 + sw, ly + 1), theme::kTrack, 1);
    if (n > 0) {
      dl->AddRectFilled(ImVec2(o.x + x0, ly - 1), ImVec2(o.x + x0 + cur, ly + 1), theme::kAccent,
                        1);
    }
    for (int k = 0; k < n; ++k) {
      const float mx = o.x + x0 + xs[static_cast<std::size_t>(k)] + kMark * 0.5F;
      const bool done = static_cast<std::size_t>(k) <= replay_;
      if (snapshot_kind_.size() > static_cast<std::size_t>(k) &&
          snapshot_kind_[static_cast<std::size_t>(k)] == EventKind::kBundleAdjustment) {
        dl->AddCircleFilled(ImVec2(mx, ly), 6.5F, theme::kSurface);
        dl->AddCircleFilled(ImVec2(mx, ly), 4.5F, done ? theme::kWarn : theme::rgb(0xFFA552, 110));
      } else {
        dl->AddRectFilled(ImVec2(mx - 1, ly - 6), ImVec2(mx + 1, ly + 6),
                          done ? theme::kTextMuted : theme::kTextDim, 1);
      }
    }
    if (n > 0) {
      dl->AddRectFilled(ImVec2(o.x + x0 + cur - 5, ly - 10), ImVec2(o.x + x0 + cur + 5, ly + 10),
                        theme::kText, 3);
    }

    // Stage-time bar and legend.
    const float by = sy + 32;
    const std::vector<double> secs(stages_.begin(), stages_.end());
    const auto frac = stage_fractions(secs);
    float bx = o.x + x0;
    const bool any = std::any_of(secs.begin(), secs.end(), [](double v) { return v > 0; });
    if (any) {
      for (std::size_t i = 0; i < frac.size(); ++i) {
        const float fw = frac[i] * (sw - 12);
        if (fw <= 0) continue;
        dl->AddRectFilled(ImVec2(bx, o.y + by), ImVec2(bx + fw, o.y + by + 6), kStageColors[i], 0);
        bx += fw + 2;
      }
    } else {
      dl->AddRectFilled(ImVec2(bx, o.y + by), ImVec2(bx + sw, o.y + by + 6), theme::kLine, 3);
    }
    float lx = o.x + x0;
    const float lyy = o.y + by + 14;
    auto legend = [&](ImU32 c, const std::string& label, bool tick) {
      if (tick) {
        dl->AddRectFilled(ImVec2(lx, lyy + 2), ImVec2(lx + 2, lyy + 12), c);
      } else {
        dl->AddRectFilled(ImVec2(lx, lyy + 3), ImVec2(lx + 8, lyy + 11), c, 2);
      }
      dl->AddText(font_set().ui, theme::kCaption, ImVec2(lx + 14, lyy), theme::kTextMuted,
                  label.c_str());
      lx += 14 + ui::text_size(font_set().ui, theme::kCaption, label.c_str()).x + 16;
    };
    int bas = 0;
    for (EventKind k : snapshot_kind_) bas += k == EventKind::kBundleAdjustment ? 1 : 0;
    if (stages_[1] > 0) legend(theme::kAccentStrong, format("features %.2f s", stages_[1]), false);
    if (stages_[2] > 0) legend(theme::kAccent, format("matching %.2f s", stages_[2]), false);
    const int runs = std::max(bas, metrics_ ? metrics_->ba_runs : 0);
    if (stages_[5] > 0) {
      legend(theme::kWarn, format("bundle adjustment %.2f s · %d runs", stages_[5], runs), false);
    } else if (runs > 0) {
      legend(theme::kWarn, format("bundle adjustment · %d runs", runs), false);
    }
    legend(theme::kTextMuted, "camera registered", true);
  }
  ImGui::End();
}

// ---------------------------------------------------------------------------------------------
// Viewport overlays

void App::viewport_overlays(const Rect& r) {
  // Toolbar (top centre).
  begin_overlay("##toolbar", ImVec2(r.x + r.w * 0.5F, r.y + 14), ImVec2(0.5F, 0.0F));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 0));
  ui::icon_button("##orbit", Icon::kOrbit, true);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("left drag: orbit · right drag: pan · wheel: zoom");
  ImGui::SameLine();
  ImGui::PushFont(font_set().ui, theme::kSmall);
  ImGui::PushStyleColor(ImGuiCol_Button, theme::rgb(0, 0));
  ImGui::PushStyleColor(ImGuiCol_Border, theme::rgb(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10, 8));
  if (ImGui::Button("Fit")) fit_view();
  ImGui::SameLine();
  if (ImGui::Button("Top")) camera_.set_angles(camera_.yaw(), 1.553F), auto_fit_ = false;
  ImGui::SameLine();
  if (ImGui::Button("Front")) camera_.set_angles(camera_.yaw(), 0.0F), auto_fit_ = false;
  ImGui::SameLine();
  const bool none = options_.selected_camera < 0;
  if (none) ImGui::BeginDisabled();
  if (ImGui::Button("Camera")) look_through_selected();
  if (none) ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip(none ? "select a camera first (click one in the view)"
                           : "look through the selected camera (C)");
  }
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(2);
  ImGui::PopFont();
  ImGui::SameLine();
  if (ui::icon_button("##shot", Icon::kCamera)) request_screenshot();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("screenshot of the window (S)");
  ImGui::SameLine();
  ImGui::PushFont(font_set().mono_medium, theme::kSmall);
  ImGui::PushStyleColor(ImGuiCol_Button, theme::rgb(0, 0));
  ImGui::PushStyleColor(ImGuiCol_Border, theme::rgb(0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(11, 8));
  if (ImGui::Button("?")) show_help_ = !show_help_;
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(2);
  ImGui::PopFont();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("keyboard and mouse (?)");
  ImGui::PopStyleVar();
  end_overlay();
  if (!status_.empty() && ImGui::GetTime() < status_until_) {  // e.g. "saved screenshot …"
    begin_overlay("##status", ImVec2(r.x + r.w * 0.5F, r.y + 64), ImVec2(0.5F, 0.0F));
    ui::text(font_set().ui, theme::kSmall, theme::kTextMuted, status_.c_str());
    end_overlay();
  }
  if (show_help_) help_overlay(r);

  const pipeline::SceneSnapshot* s = current();
  if (s == nullptr) {
    begin_overlay("##empty", ImVec2(r.x + r.w * 0.5F, r.y + r.h * 0.5F), ImVec2(0.5F, 0.5F));
    if (controller_.active()) {
      ui::text(font_set().display, theme::kHero, theme::kText, "Reconstruction starting");
      ui::text(font_set().ui, theme::kBody, theme::kTextMuted,
               "Cameras appear here as they are registered.");
    } else {
      ui::text(font_set().display, theme::kHero, theme::kText, "Nothing to show yet");
      ui::text(font_set().ui, theme::kBody, theme::kTextMuted,
               "Open a finished run from the header, or start a New run.");
    }
    end_overlay();
    return;
  }
  // LIVE badge.
  if (controller_.active()) {
    begin_overlay("##live", ImVec2(r.x + 14, r.y + 14), ImVec2(0, 0));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const char* live = "LIVE";
    const ImVec2 ts = ui::text_size(font_set().mono_medium, theme::kCaption, live);
    dl->AddRectFilled(p, ImVec2(p.x + ts.x + 26, p.y + 20), theme::kGood, 10);
    dl->AddCircleFilled(ImVec2(p.x + 10, p.y + 10), 3, theme::kOnWarn);
    dl->AddText(font_set().mono_medium, theme::kCaption, ImVec2(p.x + 18, p.y + 3), theme::kOnWarn,
                live);
    ImGui::Dummy(ImVec2(ts.x + 26, 20));
    ImGui::SameLine(0, 10);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2);
    ui::text(font_set().mono, theme::kSmall, theme::kText,
             format("snapshot %d", s->sequence).c_str());
    end_overlay();
  }
  // KPI strip (or the newest-camera card while running).
  begin_overlay("##kpi", ImVec2(r.x + 16, r.y + r.h - 16), ImVec2(0, 1));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(18, 2));
  const std::string cams = format("%zu/%zu", s->order.size(), s->ids.size());
  if (controller_.active() && !log_.empty()) {
    const RunEvent* last = nullptr;
    for (auto it = log_.rbegin(); it != log_.rend(); ++it) {
      if (it->kind == EventKind::kCamera) {
        last = &*it;
        break;
      }
    }
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p.x, p.y + 2), ImVec2(p.x + 3, p.y + 36),
                                              theme::rgb(0xFDE725), 2);
    ImGui::Dummy(ImVec2(3, 38));
    ImGui::SameLine();
    const int img = last != nullptr && last->newest_image >= 0
                        ? s->ids[static_cast<std::size_t>(last->newest_image)]
                        : -1;
    ui::kpi("NEWEST CAMERA", img >= 0 ? format("%04d", img).c_str() : "–");
    ImGui::SameLine();
    ui::kpi("CAMERAS", cams.c_str());
    ImGui::SameLine();
    ui::kpi("POINTS", thousands(static_cast<long long>(s->points.size())).c_str());
  } else {
    ui::kpi("CAMERAS", cams.c_str());
    ImGui::SameLine();
    ui::kpi("POINTS", thousands(static_cast<long long>(s->points.size())).c_str());
    std::vector<double> rot;
    for (double v : gt_.rotation_deg) {
      if (v >= 0) rot.push_back(v);
    }
    if (!rot.empty()) {
      ImGui::SameLine();
      ui::kpi("ROT MEDIAN", format("%.3f°", median(rot)).c_str());
    }
    if (metrics_ && replay_ == latest()) {
      ImGui::SameLine();
      ui::kpi("REPROJ", format("%.3f px", metrics_->reproj_median_px).c_str());
    }
  }
  ImGui::PopStyleVar();
  end_overlay();
  // Hovered point: a ring in the view and a card with its track.
  if (hovered_point_ >= 0 && static_cast<std::size_t>(hovered_point_) < s->points.size()) {
    const auto k = static_cast<std::size_t>(hovered_point_);
    const Vertex& v = geometry_.points[k];
    Eigen::Vector2f px;
    const float sx = ImGui::GetIO().DisplayFramebufferScale.x;
    if (project_to_pixel(view_proj_, Eigen::Vector3f(v.x, v.y, v.z), static_cast<int>(r.w * sx),
                         static_cast<int>(r.h * sx), px)) {
      ImGui::GetBackgroundDrawList()->AddCircle(ImVec2(r.x + px.x() / sx, r.y + px.y() / sx), 7.0F,
                                                theme::kAccent, 0, 1.5F);
    }
    begin_overlay("##hover", ImVec2(r.x + r.w - 16, r.y + r.h - 16), ImVec2(1, 1));
    ui::text(font_set().mono, theme::kCaption, theme::kTextFaint, "HOVER");
    ui::text(font_set().mono, theme::kSmall, theme::kText,
             format("point %s · %d views · %.2f px", thousands(static_cast<long long>(k)).c_str(),
                    s->point_track_length[k], s->point_error_px[k])
                 .c_str());
    ui::text(
        font_set().mono, theme::kCaption, theme::kTextMuted,
        format("(%.3f, %.3f, %.3f)", s->points[k].x(), s->points[k].y(), s->points[k].z()).c_str());
    end_overlay();
  }
}

// ---------------------------------------------------------------------------------------------

void App::request_screenshot() {
  const std::filesystem::path dir =
      (run_dir_.empty() ? std::filesystem::path("outputs") : run_dir_) / "screenshots";
  char stamp[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
  // Two shots in the same second get -2, -3, ... (the second used to overwrite the first).
  std::filesystem::path p = dir / (std::string("sfm_viewer-") + stamp + ".png");
  for (int k = 2; std::filesystem::exists(p) || p == last_screenshot_; ++k) {
    p = dir / (std::string("sfm_viewer-") + stamp + "-" + std::to_string(k) + ".png");
  }
  last_screenshot_ = p;
  screenshot_request_ = p;
}

std::filesystem::path App::take_screenshot_request() {
  std::filesystem::path p;
  std::swap(p, screenshot_request_);
  return p;
}

void App::screenshot_done(const std::filesystem::path& file, const std::string& error) {
  status_ = error.empty() ? "saved " + file.string() : "screenshot failed: " + error;
  status_until_ = ImGui::GetTime() + 4.0;
}

void App::help_overlay(const Rect& r) {
  begin_overlay("##help", ImVec2(r.x + r.w * 0.5F, r.y + r.h * 0.5F), ImVec2(0.5F, 0.5F));
  ui::text(font_set().display, theme::kHero, theme::kText, "Keyboard and mouse");
  ImGui::Dummy(ImVec2(1, 6));
  const std::pair<const char*, const char*> keys[] = {
      {"Left drag", "orbit"},
      {"Right drag", "pan"},
      {"Wheel", "zoom"},
      {"Click", "select a camera"},
      {"F", "fit the view"},
      {"C", "look through the selected camera"},
      {"Space", "play / pause replay"},
      {"Left / Right", "step through the replay"},
      {"S", "screenshot (saved under the run's screenshots/)"},
      {"Esc", "cancel the run / close this"},
      {"?", "this help"}};
  for (const auto& [k, d] : keys) {
    ui::keycap(k);
    ImGui::SameLine(110);
    ui::text(font_set().ui, theme::kSmall, theme::kTextMuted, d);
  }
  end_overlay();
}

void App::frame() {
  ImGuiIO& io = ImGui::GetIO();
  int fbw = 0;
  int fbh = 0;
  glfwGetFramebufferSize(window_, &fbw, &fbh);
  if (fbw <= 0 || fbh <= 0) return;
  poll_run();
  if (cmp_) {
    compare_frame(fbw, fbh);
    return;
  }
  if (playing_) {
    play_clock_ += io.DeltaTime;
    if (play_clock_ > 0.18) {
      play_clock_ = 0;
      if (replay_ < latest()) {
        ++replay_;
        dirty_ = true;
      } else {
        playing_ = false;
      }
    }
  }
  const Layout l = compute_layout(io.DisplaySize.x, io.DisplaySize.y);
  view_rect_ = l.viewport;
  header(l.header);
  left_rail(l.left);
  inspector(l.right);
  timeline(l.timeline);
  viewport_overlays(l.viewport);
  const float sx = io.DisplayFramebufferScale.x;
  const float sy = io.DisplayFramebufferScale.y;
  handle_input(l.viewport, sx, sy);
  if (dirty_) {
    rebuild();
  } else if (recolor_) {
    if (gpu_colors_ && renderer_->point_count() > 0) {
      try {
        geometry_.color_range =
            gpu_colors_->recolor(static_cast<int>(options_.color),
                                 {geometry_.up.x(), geometry_.up.y(), geometry_.up.z()});
        recolor_ = false;
      } catch (const std::exception& e) {
        gpu_colors_.reset();
        error_ = std::string("GPU colours off: ") + e.what();
        rebuild();
      }
    } else {
      rebuild();
    }
  }
  gl::clear_target(fbw, fbh);
  const int vx = static_cast<int>(l.viewport.x * sx);
  const int vw = static_cast<int>(l.viewport.w * sx);
  const int vh = static_cast<int>(l.viewport.h * sy);
  const int vy = fbh - static_cast<int>((l.viewport.y + l.viewport.h) * sy);
  if (vw <= 0 || vh <= 0) return;
  view_proj_ = camera_.projection(static_cast<float>(vw) / static_cast<float>(vh)) * camera_.view();
  renderer_->draw(view_proj_, point_size_ * sx, vx, vy, vw, vh);
}

}  // namespace sfm::viewer
