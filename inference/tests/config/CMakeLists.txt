sfm_add_test(test_config
  SOURCES test_parse.cpp test_validate.cpp
  LIBS sfm::config sfm::json
  LABELS unit)
if(TARGET test_config)  # skipped in CPU-only builds
  target_compile_definitions(test_config PRIVATE SFM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endif()
