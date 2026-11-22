#include "theme.hpp"

#include <string>

namespace sfm::viewer::theme {

namespace {

Fonts g_fonts;

ImFont* add(const std::filesystem::path& file) {
  if (!std::filesystem::exists(file)) return nullptr;
  ImFontConfig cfg;
  cfg.OversampleH = 2;
  return ImGui::GetIO().Fonts->AddFontFromFileTTF(file.string().c_str(), kBody, &cfg);
}

}  // namespace

ImVec4 vec(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }

Fonts load_fonts(const std::filesystem::path& dir) {
  Fonts f;
  // The UI font first: the first font added is the default one.
  f.ui = add(dir / "IBMPlexSans-Variable.ttf");
  f.mono = add(dir / "IBMPlexMono-Regular.ttf");
  f.mono_medium = add(dir / "IBMPlexMono-Medium.ttf");
  f.display = add(dir / "Gelasio-Regular.ttf");
  f.display_medium = add(dir / "Gelasio-Medium.ttf");
  f.loaded = f.ui != nullptr && f.mono != nullptr && f.mono_medium != nullptr &&
             f.display != nullptr && f.display_medium != nullptr;
  if (!f.loaded) {  // keep the layout working with the built-in font
    ImFont* d = f.ui != nullptr ? f.ui : ImGui::GetIO().Fonts->AddFontDefault();
    for (ImFont** p : {&f.ui, &f.mono, &f.mono_medium, &f.display, &f.display_medium}) {
      if (*p == nullptr) *p = d;
    }
  }
  g_fonts = f;
  return f;
}

const Fonts& fonts() { return g_fonts; }

void apply_style() {
  ImGuiStyle& s = ImGui::GetStyle();
  s.WindowPadding = ImVec2(16, 16);
  s.FramePadding = ImVec2(10, 6);
  s.ItemSpacing = ImVec2(8, 8);
  s.ItemInnerSpacing = ImVec2(8, 6);
  s.WindowRounding = 0;
  s.ChildRounding = 10;
  s.FrameRounding = 8;
  s.PopupRounding = 10;
  s.GrabRounding = 7;
  s.TabRounding = 6;
  s.ScrollbarRounding = 6;
  s.ScrollbarSize = 10;
  s.GrabMinSize = 14;
  s.WindowBorderSize = 0;
  s.ChildBorderSize = 1;
  s.FrameBorderSize = 1;
  s.PopupBorderSize = 1;
  ImVec4* c = s.Colors;
  c[ImGuiCol_Text] = vec(kText);
  c[ImGuiCol_TextDisabled] = vec(kTextFaint);
  c[ImGuiCol_WindowBg] = vec(kSurface);
  c[ImGuiCol_ChildBg] = vec(rgb(0x000000, 0));
  c[ImGuiCol_PopupBg] = vec(kRaised);
  c[ImGuiCol_Border] = vec(kLine);
  c[ImGuiCol_BorderShadow] = vec(rgb(0x000000, 0));
  c[ImGuiCol_FrameBg] = vec(kInset);
  c[ImGuiCol_FrameBgHovered] = vec(kRaised);
  c[ImGuiCol_FrameBgActive] = vec(kRaised);
  c[ImGuiCol_TitleBg] = vec(kSurface);
  c[ImGuiCol_TitleBgActive] = vec(kSurface);
  c[ImGuiCol_MenuBarBg] = vec(kSurface);
  c[ImGuiCol_ScrollbarBg] = vec(rgb(0x000000, 0));
  c[ImGuiCol_ScrollbarGrab] = vec(kTrack);
  c[ImGuiCol_ScrollbarGrabHovered] = vec(kTextDim);
  c[ImGuiCol_ScrollbarGrabActive] = vec(kTextFaint);
  c[ImGuiCol_CheckMark] = vec(kOnAccent);
  c[ImGuiCol_SliderGrab] = vec(kText);
  c[ImGuiCol_SliderGrabActive] = vec(kAccent);
  c[ImGuiCol_Button] = vec(kRaised);
  c[ImGuiCol_ButtonHovered] = vec(kTrack);
  c[ImGuiCol_ButtonActive] = vec(kLine);
  c[ImGuiCol_Header] = vec(kRaised);
  c[ImGuiCol_HeaderHovered] = vec(kTrack);
  c[ImGuiCol_HeaderActive] = vec(kLine);
  c[ImGuiCol_Separator] = vec(kLine);
  c[ImGuiCol_Tab] = vec(rgb(0x000000, 0));
  c[ImGuiCol_TabHovered] = vec(kRaised);
  c[ImGuiCol_TabSelected] = vec(rgb(0x000000, 0));
  c[ImGuiCol_TabSelectedOverline] = vec(kAccent);
  c[ImGuiCol_PlotHistogram] = vec(kBarNeutral);
  c[ImGuiCol_TextSelectedBg] = vec(rgb(0xA8C7FA, 70));
  c[ImGuiCol_NavCursor] = vec(kAccent);
}

}  // namespace sfm::viewer::theme
