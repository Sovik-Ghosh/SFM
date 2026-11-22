#pragma once
// OpenGL 3.3 core renderer for the viewer: point sprites and line segments from
// sfm::viewer::Vertex arrays. Needs a current OpenGL context for its whole life.

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <Eigen/Core>

#include "sfm/viewer/scene.hpp"

namespace sfm::viewer::gl {

class Renderer {
 public:
  Renderer();  // compiles the shaders; throws std::runtime_error on a GL error
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  void set_points(const std::vector<Vertex>& v);
  // The point vertex buffer (for CUDA-OpenGL interop: re-specified by every set_points).
  unsigned int points_vbo() const { return points_.vbo; }
  int point_count() const { return points_.count; }
  void set_lines(const std::vector<Vertex>& v);  // pairs
  // Clears to the background colour and draws lines, then points, over the whole target.
  void draw(const Eigen::Matrix4f& view_proj, float point_size_px, int width, int height);
  // The same into the rectangle (x, y) .. (x + width, y + height) of the target, in GL pixel
  // coordinates (origin bottom-left); only that rectangle is cleared.
  void draw(const Eigen::Matrix4f& view_proj, float point_size_px, int x, int y, int width,
            int height);
  // GL pixel rectangle (origin bottom-left).
  struct Rect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
  };
  // Projects into `view` but only touches pixels inside `clip` (scissor); clears clip first when
  // `clear` (overlay: the second run draws on top without clearing; wipe: two clips of one view).
  void draw(const Eigen::Matrix4f& view_proj, float point_size_px, const Rect& view,
            const Rect& clip, bool clear);
  void set_background(float r, float g, float b) { background_ = {r, g, b}; }

 private:
  struct Buffer {
    unsigned int vao = 0;
    unsigned int vbo = 0;
    int count = 0;
  };
  static void upload(Buffer& b, const std::vector<Vertex>& v);
  unsigned int program_ = 0;
  int loc_vp_ = -1;
  int loc_size_ = -1;
  int loc_round_ = -1;
  Buffer points_;
  Buffer lines_;
  std::array<float, 3> background_ = {0.0F, 0.0F, 0.0F};  // jet black (the approved design)
};

// RGB pixels (rows top-down) of the window's back buffer, after drawing and before the swap.
std::vector<std::uint8_t> read_back(int width, int height);

// Clears the whole current target (all of it, not a viewport rectangle) to black.
void clear_target(int width, int height);

// An RGB image as an OpenGL texture (the camera's input image in the inspector).
class Texture {
 public:
  Texture() = default;
  ~Texture();
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  // Replaces the texture with the image file (PPM, PNG, JPEG: whatever stb_image reads). Returns
  // false and keeps nothing if the file cannot be read.
  bool load(const std::filesystem::path& file);
  void reset();
  unsigned int id() const { return id_; }
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  unsigned int id_ = 0;
  int width_ = 0;
  int height_ = 0;
};

// Off-screen render target (RGBA8 colour + depth) for screenshots and render tests: a hidden
// window's own framebuffer cannot be read back reliably.
class Framebuffer {
 public:
  Framebuffer(int width, int height);  // throws std::runtime_error if incomplete
  ~Framebuffer();
  Framebuffer(const Framebuffer&) = delete;
  Framebuffer& operator=(const Framebuffer&) = delete;

  void bind() const;           // draw and read into this target
  static void bind_default();  // back to the window
  // RGB pixels, rows top-down (image order).
  std::vector<std::uint8_t> read_rgb() const;
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  int width_;
  int height_;
  unsigned int fbo_ = 0;
  unsigned int color_ = 0;
  unsigned int depth_ = 0;
};

}  // namespace sfm::viewer::gl
