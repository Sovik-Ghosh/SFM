// sfm_viewer: local viewer for SFM runs (OpenGL 3.3 + Dear ImGui).
//   sfm_viewer [run_dir | config.json] [--run] [--backend auto|gpu|cpu] [--size WxH]
//              [--color gray|height|error|track] [--compare RUN_B [--layout side|overlay|wipe]]
//              [--select IMAGE] [--screenshot out.png [--after-run] [--delay S] [--orbit N]]
//              [--point-size PX] [--camera-size S] [--camera-color rig|order]
//              [--zoom F] [--hidden] [--input SCRIPT]
// run_dir: an sfm_run output with stage dumps (e.g. outputs/bunny-gpu); config.json: a config to
// run live (--run starts it at once). With --screenshot the viewer renders a few frames (after the
// run ends with --after-run, or after S seconds with --delay), writes the PNG and exits (scripting,
// README images); --select shows that image's camera in the inspector; --compare opens the
// Compare screen with run_dir as A; --orbit N writes N frames out_000.png .. turning the view once
// around the model (GIF previews). --input SCRIPT replays mouse and key events (UI checks, see
// read_script) and writes the screenshots it asks for, then exits.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>

#include "app.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "theme.hpp"

namespace {

void glfw_error(int code, const char* what) {
  std::fprintf(stderr, "glfw error %d: %s\n", code, what);
}

// Scripted input for UI checks. One command per line, '#' comments; frames count from 0 and each
// command runs at the frame given by its "at N" prefix (or right after the previous one):
//   [at N] move X Y            move the mouse there
//   [at N] click X Y [right]   move there, press, release (3 frames)
//   [at N] drag X1 Y1 X2 Y2 [right]   press at 1, move to 2 in 10 steps, release
//   [at N] wheel X Y D         move there, scroll D
//   [at N] key NAME [shift]    F, C, S, Space, Escape, Left, Right, Slash
//   [at N] text STRING         characters into the focused text field
//   [at N] wait K              K frames
//   [at N] shot FILE           write the window to FILE (PNG)
struct ScriptEvent {
  enum Kind : std::uint8_t { kMove, kDown, kUp, kWheel, kKeyDown, kKeyUp, kChar, kShot };
  explicit ScriptEvent(Kind k, float px = 0, float py = 0, int b = 0)
      : kind(k), x(px), y(py), button(b) {}
  Kind kind;
  float x;
  float y;
  int button;
  ImGuiKey key = ImGuiKey_None;
  bool shift = false;
  std::string text;
};

ImGuiKey key_named(const std::string& n) {
  static const std::map<std::string, ImGuiKey> keys = {{"F", ImGuiKey_F},
                                                       {"C", ImGuiKey_C},
                                                       {"S", ImGuiKey_S},
                                                       {"Space", ImGuiKey_Space},
                                                       {"Escape", ImGuiKey_Escape},
                                                       {"Left", ImGuiKey_LeftArrow},
                                                       {"Right", ImGuiKey_RightArrow},
                                                       {"Slash", ImGuiKey_Slash},
                                                       {"Enter", ImGuiKey_Enter}};
  const auto it = keys.find(n);
  if (it == keys.end()) throw std::runtime_error("--input: unknown key " + n);
  return it->second;
}

std::map<int, std::vector<ScriptEvent>> read_script(const std::string& file, int& last_frame) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("--input: cannot open " + file);
  std::map<int, std::vector<ScriptEvent>> ev;
  int f = 5;  // after the first layout
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string cmd;
    if (!(ls >> cmd) || cmd[0] == '#') continue;
    if (cmd == "at") {
      ls >> f >> cmd;
    }
    const auto add = [&](int frame, ScriptEvent e) { ev[frame].push_back(std::move(e)); };
    if (cmd == "move") {
      float x = 0, y = 0;
      ls >> x >> y;
      add(f, ScriptEvent(ScriptEvent::kMove, x, y));
      f += 2;
    } else if (cmd == "click" || cmd == "drag") {
      float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
      ls >> x1 >> y1;
      if (cmd == "drag") ls >> x2 >> y2;
      std::string opt;
      const int b = (ls >> opt && opt == "right") ? 1 : 0;
      add(f, ScriptEvent(ScriptEvent::kMove, x1, y1));
      add(f + 1, ScriptEvent(ScriptEvent::kDown, x1, y1, b));
      int g = f + 2;
      if (cmd == "drag") {
        for (int k = 1; k <= 10; ++k, ++g) {
          const float t = static_cast<float>(k) / 10.0F;
          add(g, ScriptEvent(ScriptEvent::kMove, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t));
        }
      }
      add(g, ScriptEvent(ScriptEvent::kUp, 0, 0, b));
      f = g + 2;
    } else if (cmd == "wheel") {
      float x = 0, y = 0, d = 0;
      ls >> x >> y >> d;
      add(f, ScriptEvent(ScriptEvent::kMove, x, y));
      ScriptEvent w(ScriptEvent::kWheel, 0, d);
      add(f + 1, w);
      f += 3;
    } else if (cmd == "key") {
      std::string name, opt;
      ls >> name;
      const bool shift = static_cast<bool>(ls >> opt) && opt == "shift";
      ScriptEvent d(ScriptEvent::kKeyDown);
      d.key = key_named(name);
      d.shift = shift;
      ScriptEvent u = d;
      u.kind = ScriptEvent::kKeyUp;
      add(f, d);
      add(f + 1, u);
      f += 3;
    } else if (cmd == "text") {
      ScriptEvent t(ScriptEvent::kChar);
      std::getline(ls >> std::ws, t.text);
      add(f, t);
      f += 2;
    } else if (cmd == "wait") {
      int k = 0;
      ls >> k;
      f += k;
    } else if (cmd == "shot") {
      ScriptEvent s(ScriptEvent::kShot);
      ls >> s.text;
      add(f, s);
      f += 1;
    } else {
      throw std::runtime_error("--input: unknown command " + cmd);
    }
  }
  last_frame = f;
  return ev;
}

void apply_input(const std::vector<ScriptEvent>& events) {
  ImGuiIO& io = ImGui::GetIO();
  for (const ScriptEvent& e : events) {
    switch (e.kind) {
      case ScriptEvent::kMove:
        io.AddMousePosEvent(e.x, e.y);
        break;
      case ScriptEvent::kDown:
      case ScriptEvent::kUp:
        io.AddMouseButtonEvent(e.button, e.kind == ScriptEvent::kDown);
        break;
      case ScriptEvent::kWheel:
        io.AddMouseWheelEvent(0.0F, e.y);
        break;
      case ScriptEvent::kKeyDown:
      case ScriptEvent::kKeyUp:
        if (e.shift) io.AddKeyEvent(ImGuiMod_Shift, e.kind == ScriptEvent::kKeyDown);
        io.AddKeyEvent(e.key, e.kind == ScriptEvent::kKeyDown);
        break;
      case ScriptEvent::kChar:
        io.AddInputCharactersUTF8(e.text.c_str());
        break;
      case ScriptEvent::kShot:
        break;
    }
  }
}

}  // namespace

// The fonts: $SFM_FONT_DIR, then next to the binary (<bin>/../share/sfm/fonts for an install or a
// release archive, <bin>/fonts), then the source tree this binary was built from.
static std::string font_dir() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SFM_FONT_DIR"); env != nullptr && *env != '\0') return env;
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (!ec) {
    const fs::path bin = exe.parent_path();
    for (const fs::path& d : {bin / ".." / "share" / "sfm" / "fonts", bin / "fonts"}) {
      if (fs::is_directory(d, ec)) return d.lexically_normal().string();
    }
  }
  return SFM_FONT_DIR;
}

int main(int argc, char** argv) {
  std::string source;
  std::string screenshot;
  std::string backend = "auto";
  int width = 1440;
  int height = 900;
  bool hidden = false;
  bool run = false;
  bool after_run = false;
  double delay = 0.0;
  int select = -1;
  std::string compare;
  int color = -1;
  int hover = -1;
  bool help = false;
  bool shot_in_app = false;
  int orbit = 0;
  float camera_size = -1.0F;
  float point_size = -1.0F;
  int camera_color = -1;
  float zoom = 1.0F;
  std::string layout;
  std::string input;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--screenshot" && k + 1 < argc) {
      screenshot = argv[++k];
    } else if (a == "--size" && k + 1 < argc) {
      if (std::sscanf(argv[++k], "%dx%d", &width, &height) != 2) return 1;
    } else if (a == "--hidden") {
      hidden = true;
    } else if (a == "--run") {
      run = true;
    } else if (a == "--delay" && k + 1 < argc) {
      delay = std::atof(argv[++k]);
    } else if (a == "--hover" && k + 1 < argc) {
      hover = std::atoi(argv[++k]);
    } else if (a == "--show-help") {
      help = true;
    } else if (a == "--zoom" && k + 1 < argc) {
      zoom = static_cast<float>(std::atof(argv[++k]));
    } else if (a == "--camera-color" && k + 1 < argc) {
      const std::string c = argv[++k];
      camera_color = c == "rig" ? 1 : c == "order" ? 0 : -1;
    } else if (a == "--point-size" && k + 1 < argc) {
      point_size = static_cast<float>(std::atof(argv[++k]));
    } else if (a == "--camera-size" && k + 1 < argc) {
      camera_size = static_cast<float>(std::atof(argv[++k]));
    } else if (a == "--orbit" && k + 1 < argc) {
      orbit = std::max(0, std::atoi(argv[++k]));
    } else if (a == "--in-app-screenshot") {
      shot_in_app = true;
    } else if (a == "--color" && k + 1 < argc) {
      const std::string c = argv[++k];
      color = c == "gray" ? 0 : c == "height" ? 1 : c == "error" ? 2 : c == "track" ? 3 : -1;
    } else if (a == "--compare" && k + 1 < argc) {
      compare = argv[++k];
    } else if (a == "--layout" && k + 1 < argc) {
      layout = argv[++k];
    } else if (a == "--select" && k + 1 < argc) {
      select = std::atoi(argv[++k]);
    } else if (a == "--after-run") {
      after_run = true;
    } else if (a == "--input" && k + 1 < argc) {
      input = argv[++k];
    } else if (a == "--backend" && k + 1 < argc) {
      backend = argv[++k];
    } else if (!a.empty() && a[0] != '-') {
      source = a;
    } else {
      std::fprintf(stderr,
                   "usage: sfm_viewer [run_dir | config.json] [--run] [--backend auto|gpu|cpu] "
                   "[--color gray|height|error|track] [--compare RUN_B [--layout "
                   "side|overlay|wipe]] [--size WxH] [--select IMAGE] "
                   "[--screenshot out.png [--after-run] [--delay S] [--orbit N]] "
                   "[--point-size PX] [--camera-size S] [--camera-color rig|order] [--zoom F] "
                   "[--hidden] [--input SCRIPT]\n");
      return 1;
    }
  }
  glfwSetErrorCallback(glfw_error);
  if (glfwInit() == 0) return 1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  if (hidden) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(width, height, "SFM Viewer", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini next to the binary
  ImGui::StyleColorsDark();
  sfm::viewer::theme::load_fonts(font_dir().c_str());
  sfm::viewer::theme::apply_style();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  int status = 0;
  try {
    sfm::viewer::App app(window);
    app.set_backend(backend == "gpu"   ? sfm::viewer::BackendChoice::kGpu
                    : backend == "cpu" ? sfm::viewer::BackendChoice::kCpu
                                       : sfm::viewer::BackendChoice::kAuto);
    if (!source.empty()) {
      const bool is_config = std::filesystem::path(source).extension() == ".json";
      const bool ok =
          is_config ? app.open_config(source) && (!run || app.start_run()) : app.load(source);
      if (!ok) {
        std::fprintf(stderr, "sfm_viewer: %s\n", app.error().c_str());
        if (!screenshot.empty()) status = 1;
      }
    }
    if (select >= 0) app.select(select);
    if (hover >= 0) app.pin_hover(hover);
    if (help) app.show_help(true);
    if (camera_size > 0.0F) app.set_camera_size(camera_size);
    if (point_size > 0.0F) app.set_point_size(point_size);
    if (camera_color >= 0) app.set_color_by_camera(camera_color == 1);
    if (zoom > 0.0F && zoom != 1.0F) app.zoom_after_fit(zoom);

    if (!compare.empty() && !app.open_compare(compare)) {
      std::fprintf(stderr, "sfm_viewer: %s\n", app.error().c_str());
      if (!screenshot.empty()) status = 1;
    }
    if (!layout.empty()) app.set_compare_layout(layout == "overlay" ? 1 : layout == "wipe" ? 2 : 0);
    // Screenshots render into an off-screen target (a hidden window's framebuffer is unreadable).
    std::unique_ptr<sfm::viewer::gl::Framebuffer> offscreen;
    int script_end = -1;
    const auto script =
        input.empty() ? std::map<int, std::vector<ScriptEvent>>{} : read_script(input, script_end);
    if (!screenshot.empty() || !input.empty()) {
      int fw = 0;
      int fh = 0;
      glfwGetFramebufferSize(window, &fw, &fh);
      offscreen = std::make_unique<sfm::viewer::gl::Framebuffer>(fw, fh);
    }
    int frames = 0;
    int loops = 0;
    while (glfwWindowShouldClose(window) == 0 && status == 0) {
      glfwPollEvents();
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      const auto sev = script.find(loops);
      if (sev != script.end()) apply_input(sev->second);
      ImGui::NewFrame();
      if (offscreen) offscreen->bind();
      if (color >= 0 && frames == 2) app.set_color_mode(color);  // after the first build
      ++loops;
      if (shot_in_app && loops == 5) app.screenshot_now();  // overlays sized by then
      app.frame();
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      if (!input.empty()) {
        if (sev != script.end()) {
          for (const ScriptEvent& e : sev->second) {
            if (e.kind != ScriptEvent::kShot) continue;
            sfm::viewer::save_png(e.text, offscreen->width(), offscreen->height(),
                                  offscreen->read_rgb());
            std::printf("frame %d: wrote %s\n", loops - 1, e.text.c_str());
          }
        }
        if (const auto shot = app.take_screenshot_request(); !shot.empty()) {
          try {  // the in-app screenshot (button, S), from the off-screen target
            std::filesystem::create_directories(shot.parent_path());
            sfm::viewer::save_png(shot, offscreen->width(), offscreen->height(),
                                  offscreen->read_rgb());
            app.screenshot_done(shot, "");
            std::printf("frame %d: in-app screenshot %s\n", loops - 1, shot.string().c_str());
          } catch (const std::exception& e) {
            app.screenshot_done(shot, e.what());
          }
        }
        sfm::viewer::gl::Framebuffer::bind_default();
        glfwSwapBuffers(window);
        if (loops > script_end) break;
        continue;
      }
      if (offscreen && !(after_run && app.running()) && glfwGetTime() >= delay && ++frames >= 5) {
        if (orbit > 0) {  // frame k of the orbit, then turn for the next
          const int k = frames - 5;
          const std::filesystem::path out(screenshot);
          char suffix[16];
          std::snprintf(suffix, sizeof(suffix), "_%03d", k);
          const auto file =
              out.parent_path() / (out.stem().string() + suffix + out.extension().string());
          sfm::viewer::save_png(file.string(), offscreen->width(), offscreen->height(),
                                offscreen->read_rgb());
          app.orbit(6.2831853F / static_cast<float>(orbit));
          if (k + 1 == orbit) {
            std::printf("wrote %d frames %s .. (%dx%d)\n", orbit, file.string().c_str(),
                        offscreen->width(), offscreen->height());
            break;
          }
        } else {
          sfm::viewer::save_png(screenshot, offscreen->width(), offscreen->height(),
                                offscreen->read_rgb());
          std::printf("wrote %s (%dx%d, point colours on the %s)\n", screenshot.c_str(),
                      offscreen->width(), offscreen->height(), app.gpu_colors() ? "GPU" : "CPU");
          break;
        }
      }
      if (offscreen) sfm::viewer::gl::Framebuffer::bind_default();
      if (const auto shot = app.take_screenshot_request(); !shot.empty()) {
        int fw = 0;
        int fh = 0;
        glfwGetFramebufferSize(window, &fw, &fh);
        try {
          std::filesystem::create_directories(shot.parent_path());
          sfm::viewer::save_png(shot, fw, fh, sfm::viewer::gl::read_back(fw, fh));
          app.screenshot_done(shot, "");
          if (shot_in_app) {
            std::printf("wrote %s (%dx%d, in-app)\n", shot.string().c_str(), fw, fh);
            break;
          }
        } catch (const std::exception& e) {
          app.screenshot_done(shot, e.what());
        }
      }
      glfwSwapBuffers(window);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sfm_viewer: %s\n", e.what());
    status = 1;
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return status;
}
