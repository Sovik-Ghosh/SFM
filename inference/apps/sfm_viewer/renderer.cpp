#include "renderer.hpp"

#define GL_GLEXT_PROTOTYPES
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <GL/gl.h>
#include <GL/glext.h>

#include "stb_image.h"

namespace sfm::viewer::gl {

namespace {

constexpr const char* kVertex = R"(#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
uniform mat4 u_vp;
uniform float u_size;
out vec4 v_color;
void main() {
  gl_Position = u_vp * vec4(a_pos, 1.0);
  gl_PointSize = u_size;
  v_color = a_color;
}
)";

constexpr const char* kFragment = R"(#version 330 core
in vec4 v_color;
uniform int u_round;
out vec4 frag;
void main() {
  if (u_round == 1) {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
  }
  frag = v_color;
}
)";

unsigned int compile(GLenum type, const char* src) {
  const GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (ok == 0) {
    char log[1024];
    glGetShaderInfoLog(s, sizeof log, nullptr, log);
    throw std::runtime_error(std::string("viewer shader: ") + log);
  }
  return s;
}

}  // namespace

Renderer::Renderer() {
  const GLuint vs = compile(GL_VERTEX_SHADER, kVertex);
  const GLuint fs = compile(GL_FRAGMENT_SHADER, kFragment);
  program_ = glCreateProgram();
  glAttachShader(program_, vs);
  glAttachShader(program_, fs);
  glLinkProgram(program_);
  glDeleteShader(vs);
  glDeleteShader(fs);
  GLint ok = 0;
  glGetProgramiv(program_, GL_LINK_STATUS, &ok);
  if (ok == 0) throw std::runtime_error("viewer shader: link failed");
  loc_vp_ = glGetUniformLocation(program_, "u_vp");
  loc_size_ = glGetUniformLocation(program_, "u_size");
  loc_round_ = glGetUniformLocation(program_, "u_round");
  for (Buffer* b : {&points_, &lines_}) {
    glGenVertexArrays(1, &b->vao);
    glGenBuffers(1, &b->vbo);
    glBindVertexArray(b->vao);
    glBindBuffer(GL_ARRAY_BUFFER, b->vbo);
    // OpenGL takes the attribute's byte offset in the bound buffer as a pointer argument.
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<const void*>(offsetof(Vertex, x)));  // NOLINT(performance-no-int-to-ptr)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex),
        reinterpret_cast<const void*>(offsetof(Vertex, r)));  // NOLINT(performance-no-int-to-ptr)
  }
  glBindVertexArray(0);
  if (glGetError() != GL_NO_ERROR) throw std::runtime_error("viewer renderer: GL error at setup");
}

Renderer::~Renderer() {
  for (Buffer* b : {&points_, &lines_}) {
    glDeleteBuffers(1, &b->vbo);
    glDeleteVertexArrays(1, &b->vao);
  }
  glDeleteProgram(program_);
}

void Renderer::upload(Buffer& b, const std::vector<Vertex>& v) {
  glBindBuffer(GL_ARRAY_BUFFER, b.vbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(v.size() * sizeof(Vertex)), v.data(),
               GL_DYNAMIC_DRAW);
  b.count = static_cast<int>(v.size());
}

void Renderer::set_points(const std::vector<Vertex>& v) { upload(points_, v); }
void Renderer::set_lines(const std::vector<Vertex>& v) { upload(lines_, v); }

void Renderer::draw(const Eigen::Matrix4f& view_proj, float point_size_px, int width, int height) {
  draw(view_proj, point_size_px, 0, 0, width, height);
}

void Renderer::draw(const Eigen::Matrix4f& view_proj, float point_size_px, int x, int y, int width,
                    int height) {
  draw(view_proj, point_size_px, Rect{x, y, width, height}, Rect{x, y, width, height}, true);
}

void Renderer::draw(const Eigen::Matrix4f& view_proj, float point_size_px, const Rect& view,
                    const Rect& clip, bool clear) {
  glViewport(view.x, view.y, view.w, view.h);
  glEnable(GL_SCISSOR_TEST);  // touch only the clip rectangle
  glScissor(clip.x, clip.y, clip.w, clip.h);
  if (clear) {
    glClearColor(background_[0], background_[1], background_[2], 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glUseProgram(program_);
  glUniformMatrix4fv(loc_vp_, 1, GL_FALSE, view_proj.data());
  glUniform1f(loc_size_, point_size_px);
  glUniform1i(loc_round_, 0);
  glBindVertexArray(lines_.vao);
  glDrawArrays(GL_LINES, 0, lines_.count);
  glUniform1i(loc_round_, 1);
  glBindVertexArray(points_.vao);
  glDrawArrays(GL_POINTS, 0, points_.count);
  glBindVertexArray(0);
  glUseProgram(0);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_SCISSOR_TEST);
}

std::vector<std::uint8_t> read_back(int width, int height) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                                3);
  glReadBuffer(GL_BACK);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
  // GL rows are bottom-up: flip to image order.
  const std::size_t row = static_cast<std::size_t>(width) * 3;
  for (int y = 0; y < height / 2; ++y) {
    std::swap_ranges(
        rgb.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y) * row),
        rgb.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y + 1) * row),
        rgb.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(height - 1 - y) * row));
  }
  return rgb;
}

void clear_target(int width, int height) {
  glViewport(0, 0, width, height);
  glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

Texture::~Texture() { reset(); }

void Texture::reset() {
  if (id_ != 0) glDeleteTextures(1, &id_);
  id_ = 0;
  width_ = 0;
  height_ = 0;
}

bool Texture::load(const std::filesystem::path& file) {
  reset();
  int w = 0;
  int h = 0;
  int n = 0;
  unsigned char* px = stbi_load(file.string().c_str(), &w, &h, &n, 3);
  if (px == nullptr) return false;
  glGenTextures(1, &id_);
  glBindTexture(GL_TEXTURE_2D, id_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, px);
  glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
  stbi_image_free(px);
  width_ = w;
  height_ = h;
  return true;
}

Framebuffer::Framebuffer(int width, int height) : width_(width), height_(height) {
  glGenFramebuffers(1, &fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glGenRenderbuffers(1, &color_);
  glBindRenderbuffer(GL_RENDERBUFFER, color_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_);
  glGenRenderbuffers(1, &depth_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
  const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    throw std::runtime_error("viewer: off-screen framebuffer incomplete");
  }
}

Framebuffer::~Framebuffer() {
  glDeleteRenderbuffers(1, &depth_);
  glDeleteRenderbuffers(1, &color_);
  glDeleteFramebuffers(1, &fbo_);
}

void Framebuffer::bind() const { glBindFramebuffer(GL_FRAMEBUFFER, fbo_); }
void Framebuffer::bind_default() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

std::vector<std::uint8_t> Framebuffer::read_rgb() const {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(width_) *
                               static_cast<std::size_t>(height_) * 3);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, px.data());
  // OpenGL rows go bottom-up: flip to image order.
  const std::size_t row = static_cast<std::size_t>(width_) * 3;
  for (int y = 0; y < height_ / 2; ++y) {
    std::swap_ranges(
        px.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y) * row),
        px.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(y + 1) * row),
        px.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(height_ - 1 - y) * row));
  }
  return px;
}

}  // namespace sfm::viewer::gl
