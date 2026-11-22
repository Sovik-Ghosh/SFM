sfm_add_test(test_eval
  SOURCES test_rng.cpp test_umeyama.cpp test_synthetic.cpp test_pose_error.cpp
  LIBS sfm::eval
  LABELS unit)
