# Vendored dependencies (see third_party/README.md). Exposed as sfm::<name> targets.
set(SFM_THIRD_PARTY ${PROJECT_SOURCE_DIR}/third_party)

add_library(sfm_stb INTERFACE)
target_include_directories(sfm_stb SYSTEM INTERFACE ${SFM_THIRD_PARTY}/stb)
add_library(sfm::stb ALIAS sfm_stb)

add_library(sfm_json INTERFACE)
target_include_directories(sfm_json SYSTEM INTERFACE ${SFM_THIRD_PARTY}/nlohmann)
add_library(sfm::json ALIAS sfm_json)

add_library(sfm_eigen INTERFACE)
target_include_directories(sfm_eigen SYSTEM INTERFACE ${SFM_THIRD_PARTY}/eigen)
add_library(sfm::eigen ALIAS sfm_eigen)

if(SFM_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory(${SFM_THIRD_PARTY}/googletest ${CMAKE_BINARY_DIR}/third_party/googletest EXCLUDE_FROM_ALL)
endif()

# Dear ImGui (viewer only): core + GLFW / OpenGL3 backends, built without project warnings.
if(SFM_BUILD_VIEWER)
  find_package(glfw3 3.3 CONFIG QUIET)
  find_package(OpenGL QUIET)
  if(glfw3_FOUND AND OpenGL_FOUND)
    set(imgui_dir ${SFM_THIRD_PARTY}/imgui)
    add_library(sfm_imgui STATIC
      ${imgui_dir}/imgui.cpp
      ${imgui_dir}/imgui_draw.cpp
      ${imgui_dir}/imgui_tables.cpp
      ${imgui_dir}/imgui_widgets.cpp
      ${imgui_dir}/backends/imgui_impl_glfw.cpp
      ${imgui_dir}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(sfm_imgui SYSTEM PUBLIC ${imgui_dir} ${imgui_dir}/backends)
    target_link_libraries(sfm_imgui PUBLIC glfw OpenGL::GL)
    add_library(sfm::imgui ALIAS sfm_imgui)
  else()
    message(STATUS "sfm_viewer: GLFW 3 / OpenGL not found (apt install libglfw3-dev libgl-dev); viewer not built")
    set(SFM_BUILD_VIEWER OFF)
  endif()
endif()
