sfm_add_test(test_viewer_core
  SOURCES test_colormap.cpp test_scene.cpp test_orbit_camera.cpp test_picking.cpp
          test_ground_truth.cpp test_run_controller.cpp test_layout.cpp
          test_compare.cpp
  LIBS sfm::viewer_core
  LABELS unit)

# Loading a finished bunny run equals the live final snapshot (runs the GPU pipeline once).
sfm_add_test(test_viewer_load
  SOURCES test_run_loader.cpp
  LIBS sfm::viewer_core
  LABELS slow cuda)
if(TARGET test_viewer_load)  # skipped in CPU-only builds
  target_compile_definitions(test_viewer_load PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# Off-screen render test: needs an X display (label "display": not run in CI, which has none;
# skips itself without a display).
if(SFM_BUILD_VIEWER)
  sfm_add_test(test_viewer_render
    SOURCES test_render.cpp
    LIBS sfm_viewer_gl glfw
    LABELS display)
endif()

# V5: CUDA colours written into the GL vertex buffer == the CPU scene builder's (needs a display
# and a CUDA device; label display + cuda, so not run in CI).
if(SFM_BUILD_VIEWER)
  sfm_add_test(test_viewer_interop
    SOURCES test_interop.cpp
    LIBS sfm_viewer_cuda glfw
    LABELS display cuda)
endif()
if(SFM_BUILD_VIEWER)
  sfm_add_test(test_viewer_interop_kernels
    SOURCES test_interop_kernels.cpp
    LIBS sfm_viewer_cuda
    LABELS unit cuda)
endif()
