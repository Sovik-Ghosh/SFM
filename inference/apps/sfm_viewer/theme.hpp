#pragma once
// The approved viewer design (Figma "SFM viewer — UI redesign", v2.1): jet-black colour tokens,
// fonts (Gelasio for titles and large numbers, IBM Plex Sans for UI text, IBM Plex Mono for data)
// and the ImGui style built from them.

#include <filesystem>

#include "imgui.h"

namespace sfm::viewer::theme {

// Colour tokens (the Figma variables "SFM tokens / color/*").
inline constexpr ImU32 rgb(unsigned hex, unsigned alpha = 255) {
  return IM_COL32((hex >> 16) & 255U, (hex >> 8) & 255U, hex & 255U, alpha);
}
inline constexpr ImU32 kGround = rgb(0x000000);
inline constexpr ImU32 kSurface = rgb(0x08080A);
inline constexpr ImU32 kRaised = rgb(0x111114);
inline constexpr ImU32 kInset = rgb(0x040405);
inline constexpr ImU32 kLine = rgb(0x1E1F24);
inline constexpr ImU32 kLineSoft = rgb(0x141519);
inline constexpr ImU32 kTrack = rgb(0x2A2C32);
inline constexpr ImU32 kText = rgb(0xF2F2F4);
inline constexpr ImU32 kTextMuted = rgb(0xA6A9B0);
inline constexpr ImU32 kTextFaint = rgb(0x8B8F97);
inline constexpr ImU32 kTextDim = rgb(0x55585F);
inline constexpr ImU32 kAccent = rgb(0xA8C7FA);
inline constexpr ImU32 kAccentStrong = rgb(0x6E7FA8);
inline constexpr ImU32 kOnAccent = rgb(0x0D1526);
inline constexpr ImU32 kGood = rgb(0x63D2A0);
inline constexpr ImU32 kWarn = rgb(0xFFA552);
inline constexpr ImU32 kOnWarn = rgb(0x1A0E00);
inline constexpr ImU32 kBarNeutral = rgb(0x46526C);

ImVec4 vec(ImU32 c);

struct Fonts {
  ImFont* display = nullptr;  // Gelasio Regular (serif): app name, camera id, large numbers
  ImFont* display_medium = nullptr;
  ImFont* ui = nullptr;    // IBM Plex Sans (default instance, Regular)
  ImFont* mono = nullptr;  // IBM Plex Mono Regular: values, paths, labels
  ImFont* mono_medium = nullptr;
  bool loaded = false;  // false: the files were missing, everything uses ImGui's default font
};

// Loads the fonts from `dir` (third_party/fonts) into the atlas; call after ImGui::CreateContext.
Fonts load_fonts(const std::filesystem::path& dir);
// The ImGui style for the tokens above (rounding, spacing, colours).
void apply_style();
const Fonts& fonts();

// Type scale (pixels), from the Figma text styles.
inline constexpr float kBody = 13.0F;
inline constexpr float kSmall = 12.0F;
inline constexpr float kCaption = 11.0F;
inline constexpr float kTitle = 17.0F;   // Display/Brand
inline constexpr float kHero = 26.0F;    // Display/Hero
inline constexpr float kNumber = 22.0F;  // Display/Number (stat cards)
inline constexpr float kValueL = 15.0F;  // Data/Value L

}  // namespace sfm::viewer::theme
