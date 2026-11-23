sfm_add_test(test_pipeline
  SOURCES test_registry.cpp test_stage_io.cpp test_export.cpp test_e2e_synthetic.cpp
  LIBS sfm::pipeline
  LABELS unit)
target_include_directories(test_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../recon)  # scene.hpp
if(TARGET test_pipeline)  # skipped in CPU-only builds
  target_compile_definitions(test_pipeline PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# Full bunny run (~7 s release, much longer under ASan): GPU runner only (label "slow").
sfm_add_test(test_e2e_bunny
  SOURCES test_e2e_bunny.cpp
  LIBS sfm::pipeline sfm::json
  LABELS slow)
if(TARGET test_e2e_bunny)  # skipped in CPU-only builds
  target_compile_definitions(test_e2e_bunny PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

sfm_add_test(test_e2e_bunny_cuda
  SOURCES test_e2e_bunny_cuda.cpp
  LIBS sfm::pipeline sfm::json
  LABELS slow cuda)
if(TARGET test_e2e_bunny_cuda)  # skipped in CPU-only builds
  target_compile_definitions(test_e2e_bunny_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# The bunny as a folder of renamed PNGs ("folder" layout) and the COLMAP scene export.
sfm_add_test(test_e2e_bunny_folder
  SOURCES test_e2e_bunny_folder.cpp
  LIBS sfm::pipeline sfm::stb
  LABELS slow cuda)
if(TARGET test_e2e_bunny_folder)  # skipped in CPU-only builds
  target_compile_definitions(test_e2e_bunny_folder PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# "auto" backend and progress / cancel on bunny with the GPU (fast).
sfm_add_test(test_progress
  SOURCES test_backend_gpu.cpp test_progress.cpp
  LIBS sfm::pipeline
  LABELS slow cuda)
if(TARGET test_progress)  # skipped in CPU-only builds
  target_compile_definitions(test_progress PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# The CPU fallback: CUDA sees no device, "auto" must give the CPU result byte for byte.
sfm_add_test(test_backend_nogpu
  SOURCES test_backend_nogpu.cpp
  LIBS sfm::pipeline
  LABELS slow
  ENV CUDA_VISIBLE_DEVICES=)
if(TARGET test_backend_nogpu)  # skipped in CPU-only builds
  target_compile_definitions(test_backend_nogpu PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# P9c time / memory gates on bunny (in the repository, so they run on the GPU CI job). SERIAL:
# timings must not share the GPU with other tests.
sfm_add_test(test_perf_bunny_cuda SERIAL
  SOURCES test_perf_bunny_cuda.cpp
  LIBS sfm::pipeline
  LABELS slow cuda perf)
if(TARGET test_perf_bunny_cuda)  # skipped in CPU-only builds
  target_compile_definitions(test_perf_bunny_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
