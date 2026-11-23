# Layered targets; dependencies point one way only:
#   sfm_core <- sfm_config <- sfm_data <- sfm_features <- sfm_matching <- sfm_recon <- sfm_ba <- sfm_pipeline <- apps
#   sfm_recon <- sfm_export <- sfm_pipeline
#   sfm_core <- sfm_eval (tests, tools and later the pipeline's metrics)
# GPU phases (P3-P6) add their targets next to the CPU ones and register methods.

configure_file(include/sfm/core/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/sfm/core/version.hpp @ONLY)

add_library(sfm_core
  src/core/logger.cpp
  src/core/npy.cpp
  src/core/version.cpp)
add_library(sfm::core ALIAS sfm_core)
target_include_directories(sfm_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(sfm_core PUBLIC Threads::Threads)
if(SFM_WITH_CUDA)
  target_sources(sfm_core PRIVATE
    src/core/arena.cpp
    src/core/cuda_timer.cpp
    src/core/device_info.cu
    src/core/stream.cpp)
  target_link_libraries(sfm_core PUBLIC CUDA::cudart)
  target_compile_definitions(sfm_core PUBLIC SFM_WITH_CUDA=1)
else()
  target_sources(sfm_core PRIVATE src/core/device_info_cpu.cpp)  # no devices
  target_compile_definitions(sfm_core PUBLIC SFM_WITH_CUDA=0)
endif()
sfm_set_warnings(sfm_core)

add_library(sfm_config
  src/config/parse.cpp
  src/config/validate.cpp)
add_library(sfm::config ALIAS sfm_config)
target_link_libraries(sfm_config PUBLIC sfm::core PRIVATE sfm::json)
sfm_set_warnings(sfm_config)

add_library(sfm_data
  src/data/calib.cpp
  src/data/dataset.cpp
  src/data/image_io.cpp
  src/data/image_list.cpp
  src/data/krt.cpp
  src/data/rig.cpp)
add_library(sfm::data ALIAS sfm_data)
target_link_libraries(sfm_data PUBLIC sfm::config sfm::core sfm::eigen PRIVATE sfm::stb)
sfm_set_warnings(sfm_data)

# Evaluation helpers: synthetic scenes, Umeyama alignment, pose error. Eigen only, no CUDA code.
add_library(sfm_eval
  src/eval/pose_error.cpp
  src/eval/synthetic.cpp
  src/eval/umeyama.cpp)
add_library(sfm::eval ALIAS sfm_eval)
target_link_libraries(sfm_eval PUBLIC sfm::core sfm::eigen)
sfm_set_warnings(sfm_eval)

# Feature stage (CPU): pyramid, FAST, ORB. GPU versions arrive in P3.
add_library(sfm_features
  src/features/fast.cpp
  src/features/orb.cpp
  src/features/orientation.cpp
  src/features/pyramid.cpp
  src/features/rbrief.cpp
  src/features/sift.cpp)
add_library(sfm::features ALIAS sfm_features)
target_link_libraries(sfm_features PUBLIC sfm::data)
sfm_set_warnings(sfm_features)

if(SFM_WITH_CUDA)
  # Feature stage on the GPU (P3): bit-identical to sfm_features' CPU ORB.
  add_library(sfm_features_cuda
    src/features_cuda/orb_cuda.cu
    src/features_cuda/orb_kernels.cu
    src/features_cuda/sift_cuda.cu
    src/features_cuda/sift_kernels.cu)
  add_library(sfm::features_cuda ALIAS sfm_features_cuda)
  target_link_libraries(sfm_features_cuda PUBLIC sfm::features sfm::core)
  sfm_set_warnings(sfm_features_cuda)
  # SIFT parity with the CPU needs every float operation as written: no fused multiply-adds.
  target_compile_options(sfm_features_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
endif()

# Matching stage (CPU): brute-force Hamming, F-RANSAC verification, tracks. GPU matcher: P4.
add_library(sfm_matching
  src/matching/bf_matcher.cpp
  src/matching/eight_point.cpp
  src/matching/five_point.cpp
  src/matching/ransac_f.cpp
  src/matching/tracks.cpp
  src/matching/verify_pairs.cpp)
add_library(sfm::matching ALIAS sfm_matching)
target_link_libraries(sfm_matching PUBLIC sfm::features sfm::eigen PRIVATE sfm::eval sfm::core)
# Hardware popcount for the Hamming distance (x86-64; ARM has it in the base ISA).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set_source_files_properties(src/matching/bf_matcher.cpp PROPERTIES COMPILE_OPTIONS "-mpopcnt")
endif()
sfm_set_warnings(sfm_matching)

if(SFM_WITH_CUDA)
  # Matching on the GPU (P4): batched top-2 (equal to the CPU's) and batched F-RANSAC.
  add_library(sfm_matching_cuda
    src/matching_cuda/knn_api.cpp
    src/matching_cuda/knn_cuda.cu
    src/matching_cuda/ransac_api.cpp
    src/matching_cuda/ransac_cuda.cu
    src/matching_cuda/verify_cuda.cpp)
  add_library(sfm::matching_cuda ALIAS sfm_matching_cuda)
  target_link_libraries(sfm_matching_cuda PUBLIC sfm::matching sfm::core)
  sfm_set_warnings(sfm_matching_cuda)
endif()

# Reconstruction stage (CPU): two-view geometry, triangulation, PnP, incremental loop.
add_library(sfm_recon
  src/recon/geometry.cpp
  src/recon/incremental.cpp
  src/recon/init_pair.cpp
  src/recon/pnp.cpp
  src/recon/reconstruction.cpp)
add_library(sfm::recon ALIAS sfm_recon)
target_link_libraries(sfm_recon PUBLIC sfm::matching sfm::eval sfm::eigen)
sfm_set_warnings(sfm_recon)

# Bundle adjustment (CPU): LM + Schur complement. GPU BA: P6.
add_library(sfm_ba
  src/ba/adjust.cpp
  src/ba/ba_jacobian.cpp
  src/ba/ba_problem.cpp
  src/ba/lm.cpp
  src/ba/schur_solve.cpp)
add_library(sfm::ba ALIAS sfm_ba)
target_link_libraries(sfm_ba PUBLIC sfm::recon sfm::eigen)
sfm_set_warnings(sfm_ba)

if(SFM_WITH_CUDA)
  # Bundle adjustment on the GPU: linearisation, Schur complement and Cholesky on the device.
  add_library(sfm_ba_cuda
    src/ba_cuda/ba_cuda.cpp
    src/ba_cuda/ba_device.cu)
  add_library(sfm::ba_cuda ALIAS sfm_ba_cuda)
  target_link_libraries(sfm_ba_cuda PUBLIC sfm::ba sfm::core PRIVATE CUDA::cusolver)
  sfm_set_warnings(sfm_ba_cuda)
endif()

# Export: PLY + COLMAP text.
add_library(sfm_export
  src/export/colmap_text.cpp
  src/export/ply.cpp)
add_library(sfm::export ALIAS sfm_export)
target_link_libraries(sfm_export PUBLIC sfm::recon PRIVATE sfm::stb)
sfm_set_warnings(sfm_export)

# Pipeline: method registry, stage dumps, metrics; used by apps/sfm_run.
add_library(sfm_pipeline
  src/pipeline/pipeline.cpp
  src/pipeline/registry.cpp
  src/pipeline/stage_io.cpp)
add_library(sfm::pipeline ALIAS sfm_pipeline)
target_link_libraries(sfm_pipeline PUBLIC sfm::ba sfm::export sfm::recon sfm::features sfm::config
                      sfm::data sfm::eval PRIVATE sfm::json)
if(SFM_WITH_CUDA)
  target_link_libraries(sfm_pipeline PUBLIC sfm::ba_cuda sfm::features_cuda sfm::matching_cuda)
endif()
sfm_set_warnings(sfm_pipeline)

# Viewer core: scene geometry, colours, orbit camera, picking, ground truth, run loading. No
# OpenGL (testable without a display); the GLFW / ImGui app is apps/sfm_viewer.
add_library(sfm_viewer_core
  src/viewer/colormap.cpp
  src/viewer/compare.cpp
  src/viewer/ground_truth.cpp
  src/viewer/history.cpp
  src/viewer/layout.cpp
  src/viewer/orbit_camera.cpp
  src/viewer/picking.cpp
  src/viewer/run_controller.cpp
  src/viewer/run_loader.cpp
  src/viewer/scene.cpp)
add_library(sfm::viewer_core ALIAS sfm_viewer_core)
target_link_libraries(sfm_viewer_core PUBLIC sfm::pipeline sfm::eval sfm::eigen PRIVATE sfm::json)
sfm_set_warnings(sfm_viewer_core)

add_subdirectory(apps)

if(SFM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
