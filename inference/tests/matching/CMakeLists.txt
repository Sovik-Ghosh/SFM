sfm_add_test(test_matching
  SOURCES test_bf_matcher.cpp test_fundamental.cpp test_tracks.cpp test_five_point.cpp
  LIBS sfm::matching sfm::eval
  LABELS unit)

sfm_add_test(test_matching_golden
  SOURCES test_matching_golden.cpp
  LIBS sfm::matching sfm::data
  LABELS golden)
target_compile_definitions(test_matching_golden PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# GPU matcher: top-2 parity with the CPU, batched RANSAC on synthetic data (needs a GPU).
sfm_add_test(test_matching_cuda
  SOURCES test_matching_cuda.cpp
  LIBS sfm::matching_cuda sfm::features sfm::eval
  LABELS unit cuda)
target_compile_definitions(test_matching_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
