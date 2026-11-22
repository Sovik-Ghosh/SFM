sfm_add_test(test_pipeline
  SOURCES test_registry.cpp test_stage_io.cpp test_export.cpp test_e2e_synthetic.cpp
  LIBS sfm::pipeline
  LABELS unit)
target_include_directories(test_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../recon)  # scene.hpp
target_compile_definitions(test_pipeline PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# Full bunny run (~7 s release, much longer under ASan): GPU runner only (label "slow").
sfm_add_test(test_e2e_bunny
  SOURCES test_e2e_bunny.cpp
  LIBS sfm::pipeline sfm::json
  LABELS slow)
target_compile_definitions(test_e2e_bunny PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

sfm_add_test(test_e2e_bunny_cuda
  SOURCES test_e2e_bunny_cuda.cpp
  LIBS sfm::pipeline sfm::json
  LABELS slow cuda)
target_compile_definitions(test_e2e_bunny_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# "auto" backend and progress / cancel on bunny with the GPU (fast).
sfm_add_test(test_progress
  SOURCES test_backend_gpu.cpp test_progress.cpp
  LIBS sfm::pipeline
  LABELS slow cuda)
target_compile_definitions(test_progress PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# The CPU fallback: CUDA sees no device, "auto" must give the CPU result byte for byte.
sfm_add_test(test_backend_nogpu
  SOURCES test_backend_nogpu.cpp
  LIBS sfm::pipeline
  LABELS slow
  ENV CUDA_VISIBLE_DEVICES=)
target_compile_definitions(test_backend_nogpu PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# P9c time / memory gates on bunny (in the repository, so they run on the GPU CI job). SERIAL:
# timings must not share the GPU with other tests.
sfm_add_test(test_perf_bunny_cuda SERIAL
  SOURCES test_perf_bunny_cuda.cpp
  LIBS sfm::pipeline
  LABELS slow cuda perf)
target_compile_definitions(test_perf_bunny_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
