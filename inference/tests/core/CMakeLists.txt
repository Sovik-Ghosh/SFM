sfm_add_test(test_core_version
  SOURCES test_version.cpp
  LIBS sfm::core
  LABELS unit)

sfm_add_test(test_core_error
  SOURCES test_error.cpp
  LIBS sfm::core
  LABELS unit)

sfm_add_test(test_core_logger
  SOURCES test_logger.cpp
  LIBS sfm::core
  LABELS unit)

sfm_add_test(test_core_cuda_check
  SOURCES test_cuda_check.cu
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_device_info
  SOURCES test_device_info.cpp
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_stream_timer SERIAL
  SOURCES test_stream_timer.cu
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_device_buffer SERIAL
  SOURCES test_device_buffer.cpp
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_pinned_buffer
  SOURCES test_pinned_buffer.cpp
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_arena
  SOURCES test_arena.cpp
  LIBS sfm::core
  LABELS unit cuda)

sfm_add_test(test_core_npy
  SOURCES test_npy.cpp
  LIBS sfm::core
  LABELS unit)

sfm_add_test(test_core_npy_golden
  SOURCES test_npy_golden.cpp
  LIBS sfm::core
  LABELS golden)
target_compile_definitions(test_core_npy_golden PRIVATE
  SFM_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/inference/tests/golden")

sfm_add_test(test_core_parallel
  SOURCES test_parallel.cpp
  LIBS sfm::core
  LABELS unit)
