sfm_add_test(test_recon
  SOURCES test_geometry.cpp test_pnp.cpp test_incremental.cpp test_distortion.cpp
  LIBS sfm::recon sfm::eval
  LABELS unit)
