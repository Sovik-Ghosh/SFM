#include "widgets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "sfm/viewer/colormap.hpp"

namespace sfm::viewer::ui {

namespace {

const theme::Fonts& font_set() { return theme::fonts(); }

ImVec2 add(ImVec2 a, ImVec2 b) { return ImVec2(a.x + b.x, a.y + b.y); }

}  // namespace

ImVec2 text_size(ImFont* font, float size, const char* s) {
  return font->CalcTextSizeA(size, 1e9F, 0.0F, s);
}

void text(ImFont* font, float size, ImU32 color, const char* s) {
  ImGui::PushFont(font, size);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::TextUnformatted(s);
  ImGui::PopStyleColor();
  ImGui::PopFont();
}

void text_right(ImFont* font, float size, ImU32 color, const char* s, float right_x) {
  const ImVec2 sz = text_size(font, size, s);
  ImGui::SameLine(right_x - sz.x);
  text(font, size, color, s);
}

void eyebrow(const char* s) { text(font_set().mono_medium, theme::kCaption, theme::kTextFaint, s); }

bool button(const char* label, ButtonKind kind, ImVec2 size, bool enabled) {
  ImU32 bg = theme::kRaised;
  ImU32 fg = theme::kText;
  ImU32 border = theme::kLine;
  ImU32 hover = theme::kTrack;
  switch (kind) {
    case ButtonKind::kPrimary:
      bg = theme::kAccent, fg = theme::kOnAccent, border = theme::kAccent;
      hover = theme::rgb(0xC2D7FB);
      break;
    case ButtonKind::kSecondary:
      break;
    case ButtonKind::kGhost:
      bg = theme::rgb(0, 0), hover = theme::kRaised;
      break;
    case ButtonKind::kDanger:
      bg = theme::rgb(0, 0), fg = theme::kWarn, border = theme::kWarn,
      hover = theme::rgb(0xFFA552, 30);
      break;
  }
  if (!enabled) ImGui::BeginDisabled();
  ImGui::PushStyleColor(ImGuiCol_Button, bg);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, hover);
  ImGui::PushStyleColor(ImGuiCol_Text, fg);
  ImGui::PushStyleColor(ImGuiCol_Border, border);
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14, (size.y - theme::kBody) * 0.5F - 1));
  ImGui::PushFont(font_set().ui, theme::kBody);
  const bool pressed = ImGui::Button(label, size);
  ImGui::PopFont();
  ImGui::PopStyleVar();
  ImGui::PopStyleColor(5);
  if (!enabled) ImGui::EndDisabled();
  return pressed && enabled;
}

void draw_icon(ImDrawList* dl, Icon icon, ImVec2 min, ImVec2 max, ImU32 c) {
  const ImVec2 m((min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F);
  const float s = std::min(max.x - min.x, max.y - min.y) / 16.0F;  // 16-unit icon grid
  auto P = [&](float x, float y) { return ImVec2(m.x + (x - 8) * s, m.y + (y - 8) * s); };
  const float w = 1.4F;
  switch (icon) {
    case Icon::kLogo: {
      const ImVec2 top[] = {P(2, 4.5F), P(8, 2), P(14, 4.5F), P(8, 7)};
      dl->AddPolyline(top, 4, c, ImDrawFlags_Closed, w);
      const ImVec2 box[] = {P(2, 4.5F), P(2, 11.5F), P(8, 14.5F), P(14, 11.5F), P(14, 4.5F)};
      dl->AddPolyline(box, 5, c, 0, w);
      dl->AddLine(P(8, 7), P(8, 14.5F), c, w);
      break;
    }
    case Icon::kOrbit:
      dl->AddEllipse(m, ImVec2(6.5F * s, 3.0F * s), c, 0, 0, w);
      dl->AddCircleFilled(m, 1.6F * s, c);
      break;
    case Icon::kPan:
      dl->AddLine(P(8, 1.5F), P(8, 14.5F), c, w);
      dl->AddLine(P(1.5F, 8), P(14.5F, 8), c, w);
      for (const auto& t : {std::array<float, 6>{6, 3.5F, 8, 1.5F, 10, 3.5F},
                            std::array<float, 6>{6, 12.5F, 8, 14.5F, 10, 12.5F},
                            std::array<float, 6>{3.5F, 6, 1.5F, 8, 3.5F, 10},
                            std::array<float, 6>{12.5F, 6, 14.5F, 8, 12.5F, 10}}) {
        const ImVec2 pts[] = {P(t[0], t[1]), P(t[2], t[3]), P(t[4], t[5])};
        dl->AddPolyline(pts, 3, c, 0, w);
      }
      break;
    case Icon::kCamera:
      dl->AddRect(P(1.5F, 4), P(14.5F, 13.5F), c, 2 * s, 0, w);
      dl->AddCircle(P(8, 8.7F), 2.4F * s, c, 0, w);
      dl->AddLine(P(5.5F, 4), P(6.5F, 2), c, w);
      dl->AddLine(P(6.5F, 2), P(9.5F, 2), c, w);
      dl->AddLine(P(9.5F, 2), P(10.5F, 4), c, w);
      break;
    case Icon::kBack: {
      const ImVec2 pts[] = {P(10, 3), P(5, 8), P(10, 13)};
      dl->AddPolyline(pts, 3, c, 0, 1.6F);
      break;
    }
    case Icon::kForward: {
      const ImVec2 pts[] = {P(6, 3), P(11, 8), P(6, 13)};
      dl->AddPolyline(pts, 3, c, 0, 1.6F);
      break;
    }
    case Icon::kPlay:
      dl->AddTriangleFilled(P(4.5F, 3), P(12.5F, 8), P(4.5F, 13), c);
      break;
    case Icon::kPause:
      dl->AddRectFilled(P(4.5F, 3), P(7, 13), c, 1);
      dl->AddRectFilled(P(9, 3), P(11.5F, 13), c, 1);
      break;
    case Icon::kChevron: {
      const ImVec2 pts[] = {P(4, 6), P(8, 10), P(12, 6)};
      dl->AddPolyline(pts, 3, c, 0, w);
      break;
    }
    case Icon::kSwap: {
      const ImVec2 top[] = {P(2, 5), P(13, 5), P(10, 2)};
      dl->AddPolyline(top, 3, c, 0, w);
      const ImVec2 bot[] = {P(14, 11), P(3, 11), P(6, 14)};
      dl->AddPolyline(bot, 3, c, 0, w);
      break;
    }
    case Icon::kLock:
      dl->AddRect(P(3.5F, 7), P(12.5F, 14), c, 1.5F * s, 0, 1.3F);
      dl->PathArcTo(P(8, 5), 2.5F * s, 3.14159F, 2 * 3.14159F);
      dl->PathStroke(c, 0, 1.3F);
      dl->AddLine(P(5.5F, 5), P(5.5F, 7), c, 1.3F);
      dl->AddLine(P(10.5F, 5), P(10.5F, 7), c, 1.3F);
      break;
  }
}

bool icon_button(const char* id, Icon icon, bool selected, ImU32 bg, float size) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const bool pressed = ImGui::InvisibleButton(id, ImVec2(size, size));
  const bool hovered = ImGui::IsItemHovered();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 q = add(p, ImVec2(size, size));
  if (bg != 0) {
    dl->AddRectFilled(p, q, bg, 8);
  } else if (selected || hovered) {
    dl->AddRectFilled(p, q, selected ? theme::kLine : theme::kRaised, 7);
  }
  const float pad = size * 0.25F;
  const ImU32 fg =
      bg == theme::kAccent ? theme::kOnAccent : (selected ? theme::kText : theme::kTextMuted);
  draw_icon(dl, icon, add(p, ImVec2(pad, pad)), add(q, ImVec2(-pad, -pad)),
            hovered && bg == 0 ? theme::kText : fg);
  return pressed;
}

bool segmented(const char* id, const std::vector<const char*>& options, int& selected, int columns,
               float width, int disabled) {
  ImGui::PushID(id);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const int n = static_cast<int>(options.size());
  const int rows = (n + columns - 1) / columns;
  constexpr float kPad = 3;
  constexpr float kRow = 28;
  const float cell_w =
      (width - 2 * kPad - 2.0F * static_cast<float>(columns - 1)) / static_cast<float>(columns);
  const float h = 2 * kPad + static_cast<float>(rows) * kRow + 2.0F * static_cast<float>(rows - 1);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(p, add(p, ImVec2(width, h)), theme::kInset, 8);
  dl->AddRect(p, add(p, ImVec2(width, h)), theme::kLineSoft, 8);
  bool changed = false;
  for (int i = 0; i < n; ++i) {
    const int r = i / columns;
    const int c = i % columns;
    const ImVec2 a = add(p, ImVec2(kPad + static_cast<float>(c) * (cell_w + 2),
                                   kPad + static_cast<float>(r) * (kRow + 2)));
    const ImVec2 b = add(a, ImVec2(cell_w, kRow));
    ImGui::SetCursorScreenPos(a);
    ImGui::PushID(i);
    const bool enabled = i != disabled;
    if (!enabled) ImGui::BeginDisabled();
    if (ImGui::InvisibleButton("seg", ImVec2(cell_w, kRow)) && i != selected) {
      selected = i;
      changed = true;
    }
    const bool hovered = ImGui::IsItemHovered();
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopID();
    if (i == selected) {
      dl->AddRectFilled(a, b, theme::kLine, 6);
    } else if (hovered) {
      dl->AddRectFilled(a, b, theme::kRaised, 6);
    }
    const ImVec2 ts = text_size(font_set().ui, theme::kSmall, options[static_cast<std::size_t>(i)]);
    const ImU32 col =
        !enabled ? theme::kTextDim : (i == selected ? theme::kText : theme::kTextMuted);
    dl->AddText(
        font_set().ui, theme::kSmall,
        ImVec2(std::floor((a.x + b.x - ts.x) * 0.5F), std::floor((a.y + b.y - ts.y) * 0.5F)), col,
        options[static_cast<std::size_t>(i)]);
  }
  ImGui::SetCursorScreenPos(p);
  ImGui::Dummy(ImVec2(width, h));
  ImGui::PopID();
  return changed;
}

namespace {

void draw_check(ImDrawList* dl, ImVec2 cb, bool on) {
  if (on) {
    dl->AddRectFilled(cb, add(cb, ImVec2(16, 16)), theme::kAccent, 3.5F);
    const ImVec2 pts[] = {add(cb, ImVec2(4.5F, 8.2F)), add(cb, ImVec2(6.8F, 10.5F)),
                          add(cb, ImVec2(11.5F, 5.5F))};
    dl->AddPolyline(pts, 3, theme::kOnAccent, 0, 1.6F);
  } else {
    dl->AddRect(add(cb, ImVec2(0.5F, 0.5F)), add(cb, ImVec2(15.5F, 15.5F)), theme::kTextDim, 3, 0,
                1.2F);
  }
}

}  // namespace

bool checkbox(const char* label, bool& on) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const ImVec2 ts = text_size(font_set().ui, theme::kBody, label);
  const bool pressed = ImGui::InvisibleButton(label, ImVec2(24 + ts.x, std::max(18.0F, ts.y)));
  if (pressed) on = !on;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  draw_check(dl, ImVec2(p.x, p.y + 1), on);
  dl->AddText(font_set().ui, theme::kBody,
              ImVec2(p.x + 24, p.y + std::floor((18 - ts.y) * 0.5F) + 1), theme::kText, label);
  return pressed;
}

bool layer_row(const char* id, ImU32 swatch, const char* label, const char* count, bool& on,
               float width, bool highlighted) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  constexpr float kH = 32;
  const bool pressed = ImGui::InvisibleButton(id, ImVec2(width, kH));
  if (pressed) on = !on;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  if (highlighted || ImGui::IsItemHovered()) {
    dl->AddRectFilled(p, add(p, ImVec2(width, kH)), theme::kRaised, 6);
  }
  const float cy = p.y + kH * 0.5F;
  dl->AddRectFilled(ImVec2(p.x + 8, cy - 4), ImVec2(p.x + 16, cy + 4), swatch, 2);
  const ImVec2 ls = text_size(font_set().ui, theme::kBody, label);
  dl->AddText(font_set().ui, theme::kBody, ImVec2(p.x + 26, std::floor(cy - ls.y * 0.5F)),
              theme::kText, label);
  const ImVec2 cb(p.x + width - 24, cy - 8);
  draw_check(dl, cb, on);
  if (count != nullptr && count[0] != 0) {
    const ImVec2 cs = text_size(font_set().mono, theme::kSmall, count);
    dl->AddText(font_set().mono, theme::kSmall,
                ImVec2(cb.x - 10 - cs.x, std::floor(cy - cs.y * 0.5F)), theme::kTextMuted, count);
  }
  return pressed;
}

void arrow(ImDrawList* dl, ImVec2 c, bool up, ImU32 color, float half) {
  if (up) {
    dl->AddTriangleFilled(ImVec2(c.x - half, c.y + half * 0.8F),
                          ImVec2(c.x + half, c.y + half * 0.8F), ImVec2(c.x, c.y - half * 0.9F),
                          color);
  } else {
    dl->AddTriangleFilled(ImVec2(c.x - half, c.y - half * 0.8F), ImVec2(c.x, c.y + half * 0.9F),
                          ImVec2(c.x + half, c.y - half * 0.8F), color);
  }
}

void stat_card(const char* id, const char* label, const char* value, const char* sub, Tone tone,
               float width, float height, int arrow_dir) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::PushID(id);
  ImGui::Dummy(ImVec2(width, height));
  ImGui::PopID();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRect(p, add(p, ImVec2(width, height)), theme::kLine, 10);
  const ImU32 vc = tone == Tone::kWarn ? theme::kWarn : theme::kText;
  const ImU32 sc = tone == Tone::kGood ? theme::kGood : theme::kTextMuted;
  dl->AddText(font_set().mono, theme::kCaption, add(p, ImVec2(12, 12)), theme::kTextFaint, label);
  dl->AddText(font_set().display, theme::kNumber, add(p, ImVec2(12, 28)), vc, value);
  float sx = 12;
  if (arrow_dir != 0) {
    const float ty = p.y + height - 24 + theme::kCaption * 0.55F;
    arrow(dl, ImVec2(p.x + 16, ty), arrow_dir > 0, sc);
    sx = 24;
  }
  dl->AddText(font_set().ui, theme::kCaption, add(p, ImVec2(sx, height - 24)), sc, sub);
}

void kpi(const char* label, const char* value) {
  ImGui::BeginGroup();
  text(font_set().mono, theme::kCaption, theme::kTextFaint, label);
  text(font_set().mono_medium, theme::kValueL, theme::kText, value);
  ImGui::EndGroup();
}

void keycap(const char* key) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  const ImVec2 ts = text_size(font_set().mono, theme::kCaption, key);
  const ImVec2 sz(ts.x + 10, ts.y + 4);
  ImGui::Dummy(sz);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  dl->AddRect(p, add(p, sz), theme::kLine, 4);
  dl->AddText(font_set().mono, theme::kCaption, add(p, ImVec2(5, 2)), theme::kText, key);
}

bool slider_row(const char* label, const char* value_text, float& v, float lo, float hi,
                float width) {
  const ImVec2 start = ImGui::GetCursorPos();
  text(font_set().ui, theme::kBody, theme::kText, label);
  text_right(font_set().mono, theme::kSmall, theme::kTextMuted, value_text, start.x + width);
  // Custom slider: 4 px track, accent fill, 14 px thumb.
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::PushID(label);
  ImGui::InvisibleButton("slider", ImVec2(width, 16));
  bool changed = false;
  if (ImGui::IsItemActive()) {
    const float t = std::clamp((ImGui::GetIO().MousePos.x - p.x) / width, 0.0F, 1.0F);
    const float nv = lo + t * (hi - lo);
    changed = nv != v;
    v = nv;
  }
  ImGui::PopID();
  const float t = (v - lo) / (hi - lo);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const float cy = p.y + 8;
  dl->AddRectFilled(ImVec2(p.x, cy - 2), ImVec2(p.x + width, cy + 2), theme::kLine, 2);
  dl->AddRectFilled(ImVec2(p.x, cy - 2), ImVec2(p.x + width * t, cy + 2), theme::kAccent, 2);
  dl->AddCircleFilled(ImVec2(p.x + width * t, cy), 7, theme::kText);
  return changed;
}

void panel_frame(const char* sides) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 a = ImGui::GetWindowPos();
  const ImVec2 b = add(a, ImGui::GetWindowSize());
  if (std::strchr(sides, 't') != nullptr) dl->AddLine(a, ImVec2(b.x, a.y), theme::kLine);
  if (std::strchr(sides, 'b') != nullptr) {
    dl->AddLine(ImVec2(a.x, b.y - 1), ImVec2(b.x, b.y - 1), theme::kLine);
  }
  if (std::strchr(sides, 'l') != nullptr) dl->AddLine(a, ImVec2(a.x, b.y), theme::kLine);
  if (std::strchr(sides, 'r') != nullptr) {
    dl->AddLine(ImVec2(b.x - 1, a.y), ImVec2(b.x - 1, b.y), theme::kLine);
  }
}

void gradient_bar(ImVec2 size) {
  const ImVec2 p = ImGui::GetCursorScreenPos();
  ImGui::Dummy(size);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  constexpr int kSteps = 32;
  for (int i = 0; i < kSteps; ++i) {
    const float t0 = static_cast<float>(i) / kSteps;
    const float t1 = static_cast<float>(i + 1) / kSteps;
    const Rgb c0 = viridis(t0);
    const Rgb c1 = viridis(t1);
    dl->AddRectFilledMultiColor(
        ImVec2(p.x + size.x * t0, p.y), ImVec2(p.x + size.x * t1, p.y + size.y),
        IM_COL32(c0[0], c0[1], c0[2], 255), IM_COL32(c1[0], c1[1], c1[2], 255),
        IM_COL32(c1[0], c1[1], c1[2], 255), IM_COL32(c0[0], c0[1], c0[2], 255));
  }
}

namespace {

constexpr ImGuiWindowFlags kPanelFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
constexpr ImGuiWindowFlags kOverlayFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
    ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

}  // namespace

std::string format(const char* f, ...) {
  char buf[256];
  va_list args;
  va_start(args, f);
  std::vsnprintf(buf, sizeof buf, f, args);
  va_end(args);
  return buf;
}

std::string thousands(long long v) {
  std::string s = std::to_string(v < 0 ? -v : v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) {
    s.insert(static_cast<std::size_t>(i), ",");
  }
  return v < 0 ? "-" + s : s;
}

std::string ordinal(int n) {
  const int t = n % 100;
  const char* suffix = (t >= 11 && t <= 13) ? "th"
                       : n % 10 == 1        ? "st"
                       : n % 10 == 2        ? "nd"
                       : n % 10 == 3        ? "rd"
                                            : "th";
  return std::to_string(n) + suffix;
}

double median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

bool begin_panel(const char* name, const Rect& r, ImU32 bg, ImVec2 padding) {
  ImGui::SetNextWindowPos(ImVec2(r.x, r.y));
  ImGui::SetNextWindowSize(ImVec2(r.w, r.h));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, bg);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, padding);
  const bool open = ImGui::Begin(name, nullptr, kPanelFlags);
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  return open;
}

void begin_overlay(const char* name, ImVec2 pos, ImVec2 pivot) {
  ImGui::SetNextWindowPos(pos, ImGuiCond_Always, pivot);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, theme::rgb(0x08080A, 235));
  ImGui::PushStyleColor(ImGuiCol_Border, theme::kLine);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0F);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0F);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));
  ImGui::Begin(name, nullptr, kOverlayFlags);
}

void end_overlay() {
  ImGui::End();
  ImGui::PopStyleVar(3);
  ImGui::PopStyleColor(2);
}

// Vertically centres the next item of height h in a row of height row_h starting at y0.
void center_y(float y0, float row_h, float h) {
  ImGui::SetCursorPosY(y0 + std::floor((row_h - h) * 0.5F));
}

}  // namespace sfm::viewer::ui
