sfm_add_test(test_ba
  SOURCES test_ba.cpp
  LIBS sfm::ba sfm::eval
  LABELS unit)

sfm_add_test(test_ba_cuda
  SOURCES test_ba_cuda.cpp
  LIBS sfm::ba_cuda sfm::eval
  LABELS unit cuda)

sfm_add_test(test_ba_local
  SOURCES test_adjust_local.cpp
  LIBS sfm::ba sfm::eval
  LABELS unit)

sfm_add_test(test_ba_rig
  SOURCES test_adjust_rig.cpp
  LIBS sfm::ba sfm::eval
  LABELS unit)
