sfm_add_test(test_data
  SOURCES test_image_io.cpp test_calib.cpp test_krt.cpp test_rig.cpp
  LIBS sfm::data
  LABELS unit)
target_compile_definitions(test_data PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

sfm_add_test(test_data_golden
  SOURCES test_image_io_golden.cpp test_krt_golden.cpp
  LIBS sfm::data
  LABELS golden)
target_compile_definitions(test_data_golden PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

sfm_add_test(test_data_dataset
  SOURCES test_dataset.cpp
  LIBS sfm::data
  LABELS unit cuda)
target_compile_definitions(test_data_dataset PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
