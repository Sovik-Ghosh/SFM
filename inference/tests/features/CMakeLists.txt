sfm_add_test(test_features
  SOURCES test_fast.cpp test_pyramid.cpp test_orb.cpp test_exact_math.cpp test_sift.cpp
  LIBS sfm::features sfm::matching
  LABELS unit)
if(TARGET test_features)  # skipped in CPU-only builds
  target_compile_definitions(test_features PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

sfm_add_test(test_features_golden
  SOURCES test_orb_golden.cpp
  LIBS sfm::features
  LABELS golden)
if(TARGET test_features_golden)  # skipped in CPU-only builds
  target_compile_definitions(test_features_golden PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# CPU <-> CUDA parity of the ORB detector (needs a GPU).
sfm_add_test(test_features_cuda
  SOURCES test_orb_cuda.cpp
  LIBS sfm::features_cuda sfm::eval
  LABELS unit cuda)
if(TARGET test_features_cuda)  # skipped in CPU-only builds
  target_compile_definitions(test_features_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()

# CPU <-> CUDA parity of the SIFT detector (needs a GPU).
sfm_add_test(test_sift_cuda
  SOURCES test_sift_cuda.cpp
  LIBS sfm::features_cuda
  LABELS unit cuda)
if(TARGET test_sift_cuda)  # skipped in CPU-only builds
  target_compile_definitions(test_sift_cuda PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
