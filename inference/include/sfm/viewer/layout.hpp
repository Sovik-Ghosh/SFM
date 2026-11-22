#pragma once
// Screen layout and timeline geometry of the viewer (no ImGui / OpenGL here, so it is testable):
// fixed panels around a 3D viewport, the replay scrubber's event marks, and the stage-time bar.

#include <vector>

namespace sfm::viewer {

struct Rect {
  float x = 0;
  float y = 0;
  float w = 0;
  float h = 0;
  bool contains(float px, float py) const { return px >= x && py >= y && px < x + w && py < y + h; }
};

// Panel sizes in UI units (the Figma design, 1440 x 900).
struct PanelSizes {
  float header = 52;
  float left = 248;
  float right = 320;
  float timeline = 104;
};

struct Layout {
  Rect header;    // full width, top
  Rect left;      // left rail, between header and timeline
  Rect viewport;  // the 3D view
  Rect right;     // inspector
  Rect timeline;  // full width, bottom
};

// Panels for a window of width x height. In a window too small for all panels the side panels
// shrink proportionally so the viewport keeps at least min_view units (never negative sizes).
Layout compute_layout(float width, float height, const PanelSizes& s = {}, float min_view = 160);

// x offsets (from the scrubber's left edge) of n event marks spread over `width`, marks `mark`
// wide: first at 0, last at width - mark. n == 1 -> {0}.
std::vector<float> mark_positions(int n, float width, float mark);

// Fractions of the stage-time bar per stage, summing to 1: proportional to seconds, but every
// non-zero stage gets at least min_fraction (taken from the largest stages). All zero -> equal.
std::vector<float> stage_fractions(const std::vector<double>& seconds, float min_fraction = 0.004F);

// Index of the event mark nearest to x (scrubber clicks); -1 if n == 0.
int nearest_mark(float x, int n, float width, float mark);

}  // namespace sfm::viewer
