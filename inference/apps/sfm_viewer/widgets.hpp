#pragma once
// ImGui versions of the Figma components (Components page of "SFM viewer — UI redesign"): Button,
// Segment, Layer row, Stat card, KPI, Keycap, Eyebrow and the icons, drawn with the theme tokens.

#include <cstdint>
#include <string>
#include <vector>

#include "sfm/viewer/layout.hpp"

#include "imgui.h"
#include "theme.hpp"

namespace sfm::viewer::ui {

enum class ButtonKind : std::uint8_t { kPrimary, kSecondary, kGhost, kDanger };
enum class Icon : std::uint8_t {
  kLogo,
  kOrbit,
  kPan,
  kCamera,
  kBack,
  kForward,
  kPlay,
  kPause,
  kChevron,
  kLock,
  kSwap
};

// Text in a font and size (pixels) and colour, on the current line.
void text(ImFont* font, float size, ImU32 color, const char* s);
void text_right(ImFont* font, float size, ImU32 color, const char* s, float right_x);
ImVec2 text_size(ImFont* font, float size, const char* s);
// Section label: mono medium 11 px, faint, upper case as given.
void eyebrow(const char* s);

bool button(const char* label, ButtonKind kind, ImVec2 size = ImVec2(0, 32), bool enabled = true);
// Square 32 px button with an icon; `selected` draws the raised background.
bool icon_button(const char* id, Icon icon, bool selected = false, ImU32 bg = 0, float size = 32);
// Draws an icon centred in [min, max].
void draw_icon(ImDrawList* dl, Icon icon, ImVec2 min, ImVec2 max, ImU32 color);

// Segmented control: options in rows of `columns`, one selected. Returns true when it changed.
// `disabled`: index of an option that cannot be chosen (-1: none).
bool segmented(const char* id, const std::vector<const char*>& options, int& selected, int columns,
               float width, int disabled = -1);
// Layer row: swatch, label, count, checkbox; the whole row toggles. Returns true when toggled.
bool layer_row(const char* id, ImU32 swatch, const char* label, const char* count, bool& on,
               float width, bool highlighted = false);
// Checkbox (16 px, accent when on) with a label; returns true when toggled.
bool checkbox(const char* label, bool& on);
// Stat card (bordered): label (mono), value (display serif), sub-line. Tone colours the value.
enum class Tone : std::uint8_t { kNeutral, kWarn, kGood };
// arrow: +1 draws ▲ before the sub-line, -1 ▼ (drawn, the fonts have no such glyphs).
void stat_card(const char* id, const char* label, const char* value, const char* sub, Tone tone,
               float width, float height = 84, int arrow = 0);
// A small filled triangle (▲ when up) centred at c.
void arrow(ImDrawList* dl, ImVec2 c, bool up, ImU32 color, float half = 4.0F);
// KPI: label over value, mono.
void kpi(const char* label, const char* value);
// Keycap: small bordered mono key.
void keycap(const char* key);
// Slider row: label left, value right, then a full-width slider. Returns true when moved.
bool slider_row(const char* label, const char* value_text, float& v, float lo, float hi,
                float width);

// Fills a panel window's background and draws its border lines on the given sides ("tblr").
void panel_frame(const char* sides);

// Text helpers.
std::string format(const char* f, ...);  // printf into a std::string (256 bytes)
std::string thousands(long long v);      // 5948 -> "5,948"
std::string ordinal(int n);              // 1 -> "1st"
double median(std::vector<double> v);    // -1 when empty

// A fixed panel window covering `r` (no title, no moving), background `bg`.
bool begin_panel(const char* name, const Rect& r, ImU32 bg, ImVec2 padding = ImVec2(16, 20));
// A small floating card over the 3D view at `pos` (pivot as in SetNextWindowPos); end_overlay().
void begin_overlay(const char* name, ImVec2 pos, ImVec2 pivot);
void end_overlay();
// Vertically centres the next item of height h in a row of height row_h starting at y0.
void center_y(float y0, float row_h, float h);

// Viridis gradient bar (as drawn for points and cameras).
void gradient_bar(ImVec2 size);

}  // namespace sfm::viewer::ui
