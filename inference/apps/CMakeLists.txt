# Thin front-ends over the library.
add_executable(sfm_version sfm_version.cpp)
target_link_libraries(sfm_version PRIVATE sfm::core)
sfm_set_warnings(sfm_version)

install(TARGETS sfm_version RUNTIME DESTINATION bin)

if(SFM_WITH_CUDA)  # uploads the dataset to the GPU
  add_executable(sfm_inspect sfm_inspect.cpp)
  target_link_libraries(sfm_inspect PRIVATE sfm::data)
  sfm_set_warnings(sfm_inspect)
  install(TARGETS sfm_inspect RUNTIME DESTINATION bin)
endif()

add_executable(sfm_run sfm_run.cpp)
target_link_libraries(sfm_run PRIVATE sfm::pipeline)
sfm_set_warnings(sfm_run)
install(TARGETS sfm_run RUNTIME DESTINATION bin)

# Benchmarks: not installed in the release image. bench_stages runs a config on the CPU and
# on the GPU methods and prints the time per stage (CPU-only builds: the CPU column only).
add_executable(bench_stages bench_stages.cpp)
target_link_libraries(bench_stages PRIVATE sfm::pipeline)
sfm_set_warnings(bench_stages)

if(SFM_WITH_CUDA)
  add_executable(bench_features bench_features.cpp)
  target_link_libraries(bench_features PRIVATE sfm::features_cuda)
  sfm_set_warnings(bench_features)

  add_executable(bench_matching bench_matching.cpp)
  target_link_libraries(bench_matching PRIVATE sfm::matching_cuda sfm::features_cuda)
  sfm_set_warnings(bench_matching)

  add_executable(bench_ba bench_ba.cpp)
  target_link_libraries(bench_ba PRIVATE sfm::ba_cuda sfm::eval)
  sfm_set_warnings(bench_ba)

  add_executable(bench_upload bench_upload.cpp)
  target_link_libraries(bench_upload PRIVATE sfm::core)
  sfm_set_warnings(bench_upload)
endif()

# Local viewer (OpenGL 3.3 + Dear ImGui). Built when SFM_BUILD_VIEWER is on and GLFW / OpenGL exist.
if(SFM_BUILD_VIEWER)
  # The OpenGL renderer as a library, shared by the app and its render test.
  add_library(sfm_viewer_gl sfm_viewer/renderer.cpp)
  target_include_directories(sfm_viewer_gl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sfm_viewer)
  target_link_libraries(sfm_viewer_gl PUBLIC sfm::viewer_core OpenGL::GL PRIVATE sfm::stb)
  sfm_set_warnings(sfm_viewer_gl)

  # V5: point colours written by CUDA into the GL vertex buffer. CPU-only builds: a stand-in
  # that reports "not available", so the viewer colours points on the CPU.
  if(SFM_WITH_CUDA)
    add_library(sfm_viewer_cuda sfm_viewer/interop.cu sfm_viewer/interop_layout.cpp)
    target_link_libraries(sfm_viewer_cuda PUBLIC sfm_viewer_gl sfm::core CUDA::cudart OpenGL::GL)
    set_target_properties(sfm_viewer_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  else()
    add_library(sfm_viewer_cuda sfm_viewer/interop_none.cpp)
    target_link_libraries(sfm_viewer_cuda PUBLIC sfm_viewer_gl sfm::core)
  endif()
  target_include_directories(sfm_viewer_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sfm_viewer)
  sfm_set_warnings(sfm_viewer_cuda)

  if(SFM_WITH_CUDA)
    add_executable(bench_viewer_colors bench_viewer_colors.cpp)
    target_link_libraries(bench_viewer_colors PRIVATE sfm_viewer_cuda glfw)
    sfm_set_warnings(bench_viewer_colors)
  endif()

  add_executable(sfm_viewer
    sfm_viewer/app.cpp
    sfm_viewer/compare_view.cpp
    sfm_viewer/main.cpp
    sfm_viewer/screenshot.cpp
    sfm_viewer/theme.cpp
    sfm_viewer/widgets.cpp)
  target_link_libraries(sfm_viewer PRIVATE sfm_viewer_gl sfm_viewer_cuda sfm::imgui sfm::stb sfm::json)
  # Fonts of the approved design (third_party/fonts, OFL); the app falls back to ImGui's own font
  # when the directory is missing.
  target_compile_definitions(sfm_viewer PRIVATE SFM_FONT_DIR="${PROJECT_SOURCE_DIR}/third_party/fonts")
  sfm_set_warnings(sfm_viewer)
endif()
