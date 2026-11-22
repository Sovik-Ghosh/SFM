// Compare mode of the viewer (the design's "Compare two runs" screen): two finished runs, B mapped
// into A's frame, side by side / overlaid / wiped, with their metrics, config diff and per-camera
// errors.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <map>

#include "sfm/viewer/picking.hpp"

#include "app.hpp"
#include "imgui.h"
#include "theme.hpp"
#include "widgets.hpp"

namespace sfm::viewer {

namespace {

using ui::begin_overlay;
using ui::begin_panel;
using ui::ButtonKind;
using ui::end_overlay;
using ui::format;
using ui::Icon;
using ui::thousands;
using ui::Tone;

const theme::Fonts& font_set() { return theme::fonts(); }

constexpr ImU32 kTagA = theme::rgb(0x3A3D44);
constexpr ImU32 kBarA = theme::rgb(0x4A4D55);
constexpr PanelSizes kComparePanels{52, 0, 360, 196};

// Every vertex recoloured (overlay mode: A in gray, B in the accent colour).
void tint(std::vector<Vertex>& v, ImU32 c, std::uint8_t alpha) {
  for (Vertex& x : v) {
    x.r = static_cast<std::uint8_t>(c & 255U);
    x.g = static_cast<std::uint8_t>((c >> 8) & 255U);
    x.b = static_cast<std::uint8_t>((c >> 16) & 255U);
    x.a = alpha;
  }
}

int index_of_id(const pipeline::SceneSnapshot& s, int id) {
  const auto it = std::find(s.ids.begin(), s.ids.end(), id);
  return it == s.ids.end() ? -1 : static_cast<int>(it - s.ids.begin());
}

// Small square tag "A" / "B".
void tag(const char* letter, bool b) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(20, 20));
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(p, ImVec2(p.x + 20, p.y + 20), b ? theme::kAccent : kTagA, 5);
  const ImVec2 ts = ui::text_size(font_set().ui, theme::kCaption, letter);
  dl->AddText(font_set().ui, theme::kCaption,
              ImVec2(p.x + (20 - ts.x) * 0.5F, p.y + (20 - ts.y) * 0.5F),
              b ? theme::kOnAccent : theme::kText, letter);
}

std::string run_label(const std::filesystem::path& dir) {
  const auto abs = std::filesystem::absolute(dir).lexically_normal();
  return (abs.has_filename() ? abs : abs.parent_path()).filename().string();
}

// A config value for display: "—" when absent, long numbers shortened (0.006666… -> 0.00666667).
std::string short_value(const std::string& json) {
  if (json.empty()) return "—";
  char* end = nullptr;
  const double v = std::strtod(json.c_str(), &end);
  if (end != nullptr && *end == 0 && json.size() > 10) return format("%.6g", v);
  return json;
}

}  // namespace

bool App::load_compare_side(CompareSide& side, const std::filesystem::path& dir,
                            const SnapshotAlignment* to_a) {
  side.dir = dir;
  side.run = load_run(dir);
  side.snapshot = side.run.snapshot;
  if (to_a != nullptr && to_a->valid) side.snapshot = transform_snapshot(side.snapshot, to_a->sim);
  side.gt = align_ground_truth(side.snapshot, side.run.truth);
  if (!side.run.metrics_json.empty()) side.metrics = parse_metrics(side.run.metrics_json);
  side.seconds = side.run.timings_json.empty() ? 0.0 : parse_timings(side.run.timings_json).total;
  if (!side.renderer) side.renderer = std::make_unique<gl::Renderer>();
  return true;
}

bool App::open_compare(const std::filesystem::path& b_dir) {
  if (controller_.active()) {
    error_ = "a run is in progress: cancel it first";
    return false;
  }
  if (run_dir_.empty()) {
    error_ = "open a finished run first (it becomes A)";
    return false;
  }
  try {
    auto c = std::make_unique<Compare>();
    load_compare_side(c->a, run_dir_, nullptr);
    // B: load, align onto A, reload transformed.
    load_compare_side(c->b, b_dir, nullptr);
    c->alignment = align_snapshots(c->b.snapshot, c->a.snapshot);
    if (!c->alignment.valid) {
      error_ = format("cannot align the runs: %d images registered in both (need 3)",
                      c->alignment.common);
      return false;
    }
    load_compare_side(c->b, b_dir, &c->alignment);
    c->diff = config_diff(config::to_json(c->a.run.config), config::to_json(c->b.run.config));
    c->rows = metric_rows(c->a.metrics, c->b.metrics, c->a.seconds, c->b.seconds);
    if (options_.selected_camera >= 0 && !history_.empty()) {
      const auto& s = history_.at(replay_);
      if (static_cast<std::size_t>(options_.selected_camera) < s.ids.size()) {
        c->selected_id = s.ids[static_cast<std::size_t>(options_.selected_camera)];
      }
    }
    cmp_ = std::move(c);
    compare_input_ = b_dir.string();
    fitted_ = false;
    error_.clear();
    return true;
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
}

void App::set_compare_layout(int layout) {
  if (!cmp_) return;
  cmp_->layout = static_cast<CompareLayout>(std::clamp(layout, 0, 2));
  cmp_->dirty = true;
}

void App::compare_rebuild() {
  Compare& c = *cmp_;
  for (CompareSide* side : {&c.a, &c.b}) {
    SceneOptions o = options_;
    o.ground_truth.clear();
    o.selected_camera = c.selected_id >= 0 ? index_of_id(side->snapshot, c.selected_id) : -1;
    side->geometry = build_scene(side->snapshot, o);
    std::vector<Vertex> points = show_points_ ? side->geometry.points : std::vector<Vertex>{};
    std::vector<Vertex> lines = side->geometry.lines;
    if (c.layout == CompareLayout::kOverlay) {
      const bool b = side == &c.b;
      tint(points, b ? theme::kAccent : theme::rgb(0x8B8F97), b ? 255 : 200);
      tint(lines, b ? theme::kAccent : theme::rgb(0x6B6F7A), 255);
    }
    if (show_grid_) {  // B is in A's frame: the same grid
      const auto grid = grid_lines(c.a.geometry.center, c.a.geometry.radius, c.a.geometry.up);
      lines.insert(lines.end(), grid.begin(), grid.end());
    }
    side->renderer->set_points(points);
    side->renderer->set_lines(lines);
  }
  if (!fitted_) {
    const SceneGeometry& g = c.a.geometry;
    camera_.set_up(g.up);
    camera_.fit(g.center, options_.frustums ? std::max(g.radius, g.camera_radius) : g.radius);
    c.camera_b = camera_;
    fitted_ = true;
  }
  c.dirty = false;
}

// ---------------------------------------------------------------------------------------------

void App::compare_header(const Rect& r) {
  Compare& c = *cmp_;
  if (begin_panel("##cmp_header", r, theme::kSurface, ImVec2(20, 0))) {
    ui::panel_frame("b");
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    ui::draw_icon(dl, Icon::kLogo, ImVec2(o.x + 20, o.y + 15), ImVec2(o.x + 42, o.y + 37),
                  theme::kAccent);
    ImGui::SetCursorPos(ImVec2(52, 0));
    ui::center_y(0, r.h, theme::kTitle + 4);
    ui::text(font_set().display, theme::kTitle, theme::kText, "SFM Viewer");
    ImGui::SameLine(0, 10);
    {  // COMPARE pill
      const char* t = "COMPARE";
      const ImVec2 ts = ui::text_size(font_set().mono_medium, theme::kCaption, t);
      const ImVec2 p(ImGui::GetCursorScreenPos().x, o.y + (r.h - 20) * 0.5F);
      dl->AddRectFilled(p, ImVec2(p.x + ts.x + 16, p.y + 20), theme::kRaised, 10);
      dl->AddRect(p, ImVec2(p.x + ts.x + 16, p.y + 20), theme::kLine, 10);
      dl->AddText(font_set().mono_medium, theme::kCaption,
                  ImVec2(p.x + 8, p.y + (20 - ts.y) * 0.5F), theme::kTextMuted, t);
      ImGui::Dummy(ImVec2(ts.x + 16, 1));
    }
    // A and B pickers with the swap button between them.
    for (int k = 0; k < 2; ++k) {
      const bool b = k == 1;
      ImGui::SameLine(0, b ? 4 : 18);
      ui::center_y(0, r.h, 32);
      const ImVec2 p = ImGui::GetCursorScreenPos();
      const std::string label = run_label(b ? c.b.dir : c.a.dir);
      const float w =
          38 + ui::text_size(font_set().mono_medium, theme::kBody, label.c_str()).x + 12;
      if (ImGui::InvisibleButton(b ? "##pick_b" : "##pick_a", ImVec2(w, 32)) && b) {
        ImGui::OpenPopup("pick_b");
      }
      if (!b && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("A: the run opened in the viewer; swap with the arrows");
      }
      dl->AddRectFilled(p, ImVec2(p.x + w, p.y + 32), theme::kRaised, 8);
      dl->AddRect(p, ImVec2(p.x + w, p.y + 32), theme::kLine, 8);
      ImGui::SetCursorScreenPos(ImVec2(p.x + 6, p.y + 6));
      tag(b ? "B" : "A", b);
      dl->AddText(font_set().mono_medium, theme::kBody, ImVec2(p.x + 34, p.y + 8), theme::kText,
                  label.c_str());
      ImGui::SetCursorScreenPos(ImVec2(p.x + w, p.y));
      ImGui::Dummy(ImVec2(0, 32));
      if (!b) {
        ImGui::SameLine(0, 4);
        ui::center_y(0, r.h, 28);
        if (ui::icon_button("##swap", Icon::kSwap, false, 0, 28)) {
          const std::filesystem::path a = c.a.dir;
          const std::filesystem::path bd = c.b.dir;
          const std::filesystem::path keep = run_dir_;
          run_dir_ = bd;
          if (!open_compare(a)) run_dir_ = keep;
          ImGui::End();
          return;  // cmp_ was replaced
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("swap A and B");
      }
    }
    if (ImGui::BeginPopup("pick_b")) {
      if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
      ui::eyebrow("RUN B (A FINISHED RUN DIRECTORY)");
      char buf[512];
      std::snprintf(buf, sizeof buf, "%s", compare_input_.c_str());
      ImGui::SetNextItemWidth(360);
      if (ImGui::InputText("##b", buf, sizeof buf)) compare_input_ = buf;
      ImGui::SameLine();
      if (ui::button("Compare", ButtonKind::kPrimary, ImVec2(0, 30))) {
        const std::string in = compare_input_;
        if (open_compare(in)) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        ImGui::End();
        return;
      }
      if (!error_.empty()) ui::text(font_set().ui, theme::kSmall, theme::kWarn, error_.c_str());
      ImGui::EndPopup();
    }
    // Right side: layout, link, done.
    float x = r.w - 16;
    auto place = [&](float w) {
      x -= w;
      ImGui::SameLine(x);
    };
    place(ui::text_size(font_set().ui, theme::kBody, "Done").x + 28);
    ui::center_y(0, r.h, 32);
    if (ui::button("Done", ButtonKind::kGhost)) {
      ImGui::End();
      close_compare();
      fitted_ = false;
      dirty_ = true;
      return;
    }
    place(24 + ui::text_size(font_set().ui, theme::kBody, "Link cameras").x + 16);
    ui::center_y(0, r.h, 18);
    if (ui::checkbox("Link cameras", c.link) && c.link) c.camera_b = camera_;
    place(270 + 16);
    ui::center_y(0, r.h, 34);
    int layout = static_cast<int>(c.layout);
    if (ui::segmented("layout", {"Side by side", "Overlay", "Wipe"}, layout, 3, 270)) {
      c.layout = static_cast<CompareLayout>(layout);
      c.dirty = true;
    }
  }
  ImGui::End();
}

void App::compare_panel(const Rect& r) {
  Compare& c = *cmp_;
  if (begin_panel("##cmp_panel", r, theme::kSurface, ImVec2(16, 20))) {
    ui::panel_frame("l");
    const float w = r.w - 32;
    const float x0 = ImGui::GetCursorPosX();
    ui::eyebrow("B AGAINST A");
    ImGui::Dummy(ImVec2(1, 2));
    const auto pct = [](double a, double b) {
      return a != 0 ? 100.0 * (b - a) / std::fabs(a) : 0.0;
    };
    const float cw = (w - 8) * 0.5F;
    const MetricRow* points = nullptr;
    const MetricRow* rot = nullptr;
    for (const MetricRow& m : c.rows) {
      if (m.name == "Points") points = &m;
      if (m.name == "Rotation median °") rot = &m;
    }
    auto card = [&](const char* id, const char* label, const MetricRow* m, const std::string& value,
                    int decimals_pct) {
      (void)decimals_pct;
      if (m == nullptr || !m->valid) {
        ui::stat_card(id, label, "–", "", Tone::kNeutral, cw);
        return;
      }
      const int v = verdict(*m);
      const double d = pct(m->a, m->b);
      const std::string sub = format("%.0f %%", std::fabs(d));
      ui::stat_card(id, label, value.c_str(), sub.c_str(),
                    v > 0 ? Tone::kGood : (v < 0 ? Tone::kWarn : Tone::kNeutral), cw, 84,
                    std::fabs(d) < 0.5 ? 0 : (d > 0 ? 1 : -1));
    };
    card("k1", "POINTS", points, points ? thousands(static_cast<long long>(points->b)) : "", 0);
    ImGui::SameLine(0, 8);
    card("k2", "ROT MEDIAN", rot, rot && rot->valid ? format("%.3f°", rot->b) : "", 0);

    // Metric table.
    ImGui::Dummy(ImVec2(1, 6));
    const float ca = x0 + w - 76 - 64 - 64;  // right edges of the A, B, delta columns
    const float cb = x0 + w - 76 - 4;
    const float cd = x0 + w;
    ui::text(font_set().mono, theme::kCaption, theme::kTextFaint, "METRIC");
    ui::text_right(font_set().mono, theme::kCaption, theme::kTextFaint, "A", ca + 60);
    ui::text_right(font_set().mono, theme::kCaption, theme::kTextFaint, "B", cb);
    ui::text_right(font_set().mono, theme::kCaption, theme::kTextFaint, "CHANGE", cd);
    for (const MetricRow& m : c.rows) {
      const ImVec2 p = ImGui::GetCursorScreenPos();
      ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y - 3), ImVec2(p.x + w, p.y - 3),
                                          theme::kLineSoft);
      ui::text(font_set().ui, theme::kSmall, theme::kText, m.name.c_str());
      if (!m.valid) {
        ui::text_right(font_set().mono, theme::kSmall, theme::kTextFaint, "n/a", cd);
        continue;
      }
      const std::string fa =
          m.decimals == 0 ? thousands(std::llround(m.a)) : format("%.*f", m.decimals, m.a);
      const std::string fb =
          m.decimals == 0 ? thousands(std::llround(m.b)) : format("%.*f", m.decimals, m.b);
      const int v = verdict(m);
      const double d = m.b - m.a;
      std::string fd;
      const bool same = std::fabs(d) * std::pow(10.0, m.decimals) < 0.5;
      if (same) {
        fd = "=";
      } else if (m.decimals == 0 && std::fabs(m.a) >= 100) {
        fd = format("%.0f %%", std::fabs(pct(m.a, m.b)));
      } else {
        fd = format("%.*f", m.decimals, std::fabs(d));
      }
      const ImU32 col = v > 0 ? theme::kGood : (v < 0 ? theme::kWarn : theme::kTextFaint);
      ui::text_right(font_set().mono, theme::kSmall, theme::kTextMuted, fa.c_str(), ca + 60);
      ui::text_right(font_set().mono, theme::kSmall, theme::kText, fb.c_str(), cb);
      ui::text_right(font_set().mono, theme::kSmall, col, fd.c_str(), cd);
      if (!same) {
        const ImVec2 q = ImGui::GetItemRectMin();
        ui::arrow(ImGui::GetWindowDrawList(), ImVec2(q.x - 8, q.y + theme::kSmall * 0.55F), d > 0,
                  col, 3.5F);
      }
    }
    {
      const ImVec2 q = ImGui::GetCursorScreenPos();
      ImDrawList* dl = ImGui::GetWindowDrawList();
      ui::arrow(dl, ImVec2(q.x + 4, q.y + 7), true, theme::kTextFaint, 3.5F);
      ui::arrow(dl, ImVec2(q.x + 13, q.y + 7), false, theme::kTextFaint, 3.5F);
      ImGui::SetCursorScreenPos(ImVec2(q.x + 22, q.y));
      ui::text(font_set().ui, theme::kCaption, theme::kTextFaint,
               "direction of change · mint better · orange worse");
    }

    // Config diff.
    ImGui::Dummy(ImVec2(1, 10));
    ui::eyebrow(format("CONFIG.JSON DIFF · %zu KEY%s", c.diff.size(), c.diff.size() == 1 ? "" : "S")
                    .c_str());
    ImGui::Dummy(ImVec2(1, 2));
    const float left = r.h - ImGui::GetCursorPosY() - 20;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::kInset);
    ImGui::PushStyleColor(ImGuiCol_Border, theme::kLineSoft);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
    if (ImGui::BeginChild("##diff", ImVec2(w, std::max(left, 60.0F)), ImGuiChildFlags_Borders)) {
      if (c.diff.empty()) {
        ui::text(font_set().mono, theme::kSmall, theme::kTextFaint, "identical configs");
      }
      for (const ConfigDifference& d : c.diff) {
        ui::text(font_set().mono, theme::kSmall, theme::kTextMuted, d.key.c_str());
        ui::text(font_set().mono, theme::kSmall, theme::kWarn, ("− " + short_value(d.a)).c_str());
        ImGui::SameLine(0, 14);
        ui::text(font_set().mono, theme::kSmall, theme::kGood, ("+ " + short_value(d.b)).c_str());
        ImGui::Dummy(ImVec2(1, 2));
      }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
  }
  ImGui::End();
}

void App::compare_chart(const Rect& r) {
  Compare& c = *cmp_;
  if (begin_panel("##cmp_chart", r, theme::kSurface, ImVec2(20, 16))) {
    ui::panel_frame("t");
    const float w = r.w - 40;
    // Images with a rotation error in both runs, in dataset order.
    struct Pair {
      int id;
      double a;
      double b;
    };
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < c.a.snapshot.ids.size(); ++i) {
      if (!c.a.gt.valid || c.a.gt.rotation_deg[i] < 0) continue;
      const int j = index_of_id(c.b.snapshot, c.a.snapshot.ids[i]);
      if (j < 0 || !c.b.gt.valid || c.b.gt.rotation_deg[static_cast<std::size_t>(j)] < 0) continue;
      pairs.push_back({c.a.snapshot.ids[i], c.a.gt.rotation_deg[i],
                       c.b.gt.rotation_deg[static_cast<std::size_t>(j)]});
    }
    int lower = 0;
    double top = 1e-9;
    for (const Pair& p : pairs) {
      lower += p.b < p.a ? 1 : 0;
      top = std::max({top, p.a, p.b});
    }
    top *= 1.1;
    ui::eyebrow("ROTATION ERROR PER CAMERA");
    ImGui::SameLine(0, 18);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    for (const auto& [col, name] : {std::pair{kBarA, "A"}, std::pair{theme::kAccent, "B"}}) {
      const ImVec2 p = ImGui::GetCursorScreenPos();
      dl->AddRectFilled(ImVec2(p.x, p.y + 3), ImVec2(p.x + 10, p.y + 13), col, 2);
      ImGui::SetCursorScreenPos(ImVec2(p.x + 16, p.y));
      ui::text(font_set().ui, theme::kSmall, theme::kTextMuted, name);
      ImGui::SameLine(0, 14);
    }
    const std::string summary =
        pairs.empty()
            ? std::string("no ground truth in one of the runs")
            : format("B lower on %d / %zu cameras · click a pair to select it in both views", lower,
                     pairs.size());
    ui::text_right(font_set().ui, theme::kSmall, theme::kTextMuted, summary.c_str(), 20 + w);
    if (pairs.empty()) {
      ImGui::End();
      return;
    }
    ImGui::Dummy(ImVec2(1, 4));
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    constexpr float kAxis = 40;
    const float plot_h = r.h - (p0.y - ImGui::GetWindowPos().y) - 16 - 18;
    const float pw = w - kAxis;
    for (const double value : {top / 1.1, top / 2.2, 0.0}) {
      const std::string label = value > 0 ? format("%.2g°", value) : std::string("0");
      const float y = p0.y + plot_h * static_cast<float>(1.0 - value / top);
      const ImVec2 ts = ui::text_size(font_set().mono, theme::kCaption, label.c_str());
      dl->AddText(font_set().mono, theme::kCaption,
                  ImVec2(p0.x + kAxis - 8 - ts.x, y - ts.y * 0.5F), theme::kTextFaint,
                  label.c_str());
    }
    const float gap = pairs.size() > 60 ? 2.0F : 8.0F;
    const float gw =
        (pw - gap * static_cast<float>(pairs.size() - 1)) / static_cast<float>(pairs.size());
    ImGui::SetCursorScreenPos(ImVec2(p0.x + kAxis, p0.y));
    ImGui::InvisibleButton("##pairs", ImVec2(pw, plot_h));
    const bool clicked = ImGui::IsItemClicked();
    const bool hovered = ImGui::IsItemHovered();
    const float mx = ImGui::GetIO().MousePos.x - (p0.x + kAxis);
    for (std::size_t k = 0; k < pairs.size(); ++k) {
      const float gx = p0.x + kAxis + static_cast<float>(k) * (gw + gap);
      const float bw = std::max((gw - 1) * 0.5F, 1.0F);
      const float ha = std::max(2.0F, static_cast<float>(pairs[k].a / top) * plot_h);
      const float hb = std::max(2.0F, static_cast<float>(pairs[k].b / top) * plot_h);
      const bool sel = pairs[k].id == c.selected_id;
      if (sel) {
        dl->AddRectFilled(ImVec2(gx - 2, p0.y), ImVec2(gx + gw + 2, p0.y + plot_h), theme::kRaised,
                          3);
      }
      dl->AddRectFilled(ImVec2(gx, p0.y + plot_h - ha), ImVec2(gx + bw, p0.y + plot_h), kBarA, 2,
                        ImDrawFlags_RoundCornersTop);
      dl->AddRectFilled(ImVec2(gx + bw + 1, p0.y + plot_h - hb),
                        ImVec2(gx + 2 * bw + 1, p0.y + plot_h), theme::kAccent, 2,
                        ImDrawFlags_RoundCornersTop);
      const bool under =
          mx >= gx - (p0.x + kAxis) - gap * 0.5F && mx < gx - (p0.x + kAxis) + gw + gap * 0.5F;
      if (under && hovered) {
        ImGui::SetTooltip("%04d   A %.3f°   B %.3f°", pairs[k].id, pairs[k].a, pairs[k].b);
        if (clicked) {
          c.selected_id = pairs[k].id;
          c.dirty = true;
        }
      }
    }
    dl->AddLine(ImVec2(p0.x + kAxis, p0.y + plot_h), ImVec2(p0.x + kAxis + pw, p0.y + plot_h),
                theme::kLine);
    const int ticks = std::min<int>(5, static_cast<int>(pairs.size()));
    for (int t = 0; t < ticks; ++t) {
      const std::size_t k = ticks == 1 ? 0
                                       : static_cast<std::size_t>(t) * (pairs.size() - 1) /
                                             static_cast<std::size_t>(ticks - 1);
      const std::string l = format("%04d", pairs[k].id);
      const ImVec2 ts = ui::text_size(font_set().mono, theme::kCaption, l.c_str());
      float x = p0.x + kAxis + static_cast<float>(k) * (gw + gap) + gw * 0.5F - ts.x * 0.5F;
      x = std::clamp(x, p0.x + kAxis, p0.x + kAxis + pw - ts.x);
      dl->AddText(font_set().mono, theme::kCaption, ImVec2(x, p0.y + plot_h + 6), theme::kTextFaint,
                  l.c_str());
    }
  }
  ImGui::End();
}

void App::compare_labels(const Rect& view) {
  Compare& c = *cmp_;
  auto label = [&](const char* id, const CompareSide& s, bool b, ImVec2 pos) {
    begin_overlay(id, pos, ImVec2(0, 0));
    tag(b ? "B" : "A", b);
    ImGui::SameLine(0, 10);
    ImGui::BeginGroup();
    ui::text(font_set().mono_medium, theme::kBody, theme::kText, s.run.config.run.name.c_str());
    std::vector<double> rot;
    for (double v : s.gt.rotation_deg) {
      if (v >= 0) rot.push_back(v);
    }
    const std::string sub =
        rot.empty()
            ? format("%s pts", thousands(static_cast<long long>(s.snapshot.points.size())).c_str())
            : format("%s pts · %.3f°",
                     thousands(static_cast<long long>(s.snapshot.points.size())).c_str(),
                     ui::median(rot));
    ui::text(font_set().mono, theme::kSmall, theme::kTextMuted, sub.c_str());
    ImGui::EndGroup();
    end_overlay();
  };
  if (c.layout == CompareLayout::kSideBySide) {
    label("##lab_a", c.a, false, ImVec2(view.x + 14, view.y + 14));
    label("##lab_b", c.b, true, ImVec2(view.x + view.w * 0.5F + 15, view.y + 14));
  } else {
    label("##lab_a", c.a, false, ImVec2(view.x + 14, view.y + 14));
    label("##lab_b", c.b, true, ImVec2(view.x + view.w - 14 - 230, view.y + 14));
  }
  if (!c.alignment.valid) return;
  begin_overlay("##align", ImVec2(view.x + 14, view.y + view.h - 14), ImVec2(0, 1));
  ui::text(
      font_set().mono, theme::kCaption, theme::kTextFaint,
      format("B aligned to A: %d shared cameras, scale %.3f", c.alignment.common, c.alignment.sim.s)
          .c_str());
  end_overlay();
}

void App::compare_input(const Rect& view, float sx, float sy) {
  Compare& c = *cmp_;
  ImGuiIO& io = ImGui::GetIO();
  if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F)) {
    fitted_ = false;
    c.dirty = true;
  }
  if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    close_compare();
    fitted_ = false;
    dirty_ = true;
    return;
  }
  if (io.WantCaptureMouse) return;
  const ImVec2 m = io.MousePos;
  const ImVec2 click = io.MouseClickedPos[0];
  const float divider = view.x + view.w * (c.layout == CompareLayout::kWipe ? c.wipe : 0.5F);
  // Wipe divider: drag it.
  if (c.layout == CompareLayout::kWipe) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && std::fabs(m.x - divider) < 8 &&
        view.contains(m.x, m.y)) {
      c.dragging_wipe = true;
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) c.dragging_wipe = false;
    if (c.dragging_wipe) {
      c.wipe = std::clamp((m.x - view.x) / view.w, 0.02F, 0.98F);
      return;
    }
  }
  const bool right_side = c.layout == CompareLayout::kSideBySide && click.x >= divider;
  OrbitCamera& cam = (right_side && !c.link) ? c.camera_b : camera_;
  const ImVec2 d = io.MouseDelta;
  if (view.contains(click.x, click.y) && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    cam.rotate(-d.x * 0.006F, d.y * 0.006F);
    if (c.link) c.camera_b = camera_;
  }
  const ImVec2 rc = io.MouseClickedPos[1];
  if (view.contains(rc.x, rc.y) && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    cam.pan(-d.x / view.h, d.y / view.h);
    if (c.link) c.camera_b = camera_;
  }
  if (!view.contains(m.x, m.y)) return;
  if (io.MouseWheel != 0.0F) {
    OrbitCamera& zc = (c.layout == CompareLayout::kSideBySide && m.x >= divider && !c.link)
                          ? c.camera_b
                          : camera_;
    zc.zoom(std::pow(0.88F, io.MouseWheel));
    if (c.link) c.camera_b = camera_;
  }
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && view.contains(click.x, click.y) &&
      ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0F).x == 0.0F &&
      ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0F).y == 0.0F) {
    // Pick in the half that was clicked (side by side) or in A's cameras (overlay / wipe).
    Rect r = view;
    const CompareSide* s = &c.a;
    const OrbitCamera* pc = &camera_;
    if (c.layout == CompareLayout::kSideBySide) {
      r.w = view.w * 0.5F;
      if (m.x >= divider) {
        r.x = divider;
        s = &c.b;
        pc = c.link ? &camera_ : &c.camera_b;
      }
    } else if (c.layout == CompareLayout::kWipe && m.x >= divider) {
      s = &c.b;
    }
    const int w = static_cast<int>(r.w * sx);
    const int h = static_cast<int>(r.h * sy);
    const Eigen::Matrix4f vp =
        pc->projection(static_cast<float>(w) / static_cast<float>(h)) * pc->view();
    const int i = pick_camera(s->geometry.camera_centres, vp, w, h,
                              Eigen::Vector2f((m.x - r.x) * sx, (m.y - r.y) * sy));
    c.selected_id = i >= 0 ? s->snapshot.ids[static_cast<std::size_t>(i)] : -1;
    c.dirty = true;
  }
}

void App::compare_frame(int fbw, int fbh) {
  ImGuiIO& io = ImGui::GetIO();
  const Layout l = compute_layout(io.DisplaySize.x, io.DisplaySize.y, kComparePanels);
  compare_header(l.header);
  if (!cmp_) return;  // Done / swap closed or replaced it
  compare_panel(l.right);
  compare_chart(l.timeline);
  compare_labels(l.viewport);
  const float sx = io.DisplayFramebufferScale.x;
  const float sy = io.DisplayFramebufferScale.y;
  compare_input(l.viewport, sx, sy);
  if (!cmp_) return;
  Compare& c = *cmp_;
  if (c.dirty || !fitted_) compare_rebuild();
  gl::clear_target(fbw, fbh);
  const Rect& v = l.viewport;
  auto to_gl = [&](float x, float y, float w, float h) {
    return gl::Renderer::Rect{static_cast<int>(x * sx), fbh - static_cast<int>((y + h) * sy),
                              static_cast<int>(w * sx), static_cast<int>(h * sy)};
  };
  auto vp_of = [](const OrbitCamera& cam, const gl::Renderer::Rect& r) {
    return Eigen::Matrix4f(
        cam.projection(static_cast<float>(r.w) / static_cast<float>(std::max(r.h, 1))) *
        cam.view());
  };
  const float ps = point_size_ * sx;
  const OrbitCamera& cam_b = c.link ? camera_ : c.camera_b;
  switch (c.layout) {
    case CompareLayout::kSideBySide: {
      const float half = std::floor(v.w * 0.5F);
      const auto ra = to_gl(v.x, v.y, half - 1, v.h);
      const auto rb = to_gl(v.x + half + 1, v.y, v.w - half - 1, v.h);
      c.a.renderer->draw(vp_of(camera_, ra), ps, ra, ra, true);
      c.b.renderer->draw(vp_of(cam_b, rb), ps, rb, rb, true);
      ImGui::GetBackgroundDrawList()->AddRectFilled(
          ImVec2(v.x + half - 1, v.y), ImVec2(v.x + half + 1, v.y + v.h), theme::kLine);
      break;
    }
    case CompareLayout::kOverlay: {
      const auto r = to_gl(v.x, v.y, v.w, v.h);
      c.a.renderer->draw(vp_of(camera_, r), ps, r, r, true);
      c.b.renderer->draw(vp_of(camera_, r), ps, r, r, false);
      break;
    }
    case CompareLayout::kWipe: {
      const auto r = to_gl(v.x, v.y, v.w, v.h);
      const float split = std::floor(v.w * c.wipe);
      const auto left = to_gl(v.x, v.y, split, v.h);
      const auto right = to_gl(v.x + split, v.y, v.w - split, v.h);
      c.a.renderer->draw(vp_of(camera_, r), ps, r, left, true);
      c.b.renderer->draw(vp_of(camera_, r), ps, r, right, true);
      ImDrawList* fg = ImGui::GetBackgroundDrawList();
      const float x = v.x + split;
      fg->AddLine(ImVec2(x, v.y), ImVec2(x, v.y + v.h), theme::kText, 2.0F);
      fg->AddCircleFilled(ImVec2(x, v.y + v.h * 0.5F), 10, theme::kText);
      ui::draw_icon(fg, Icon::kSwap, ImVec2(x - 6, v.y + v.h * 0.5F - 6),
                    ImVec2(x + 6, v.y + v.h * 0.5F + 6), theme::kGround);
      break;
    }
  }
}

}  // namespace sfm::viewer
