#include <gtest/gtest.h>

#include "sfm/core/cuda_check.h"
#include "sfm/core/device_info.hpp"

TEST(DeviceInfo, AtLeastOneDevice) { EXPECT_GE(sfm::core::device_count(), 1); }

TEST(DeviceInfo, PropertiesAreSane) {
  const auto info = sfm::core::query_device(0);
  EXPECT_FALSE(info.name.empty());
  EXPECT_GE(info.cc_major, 7);  // CUDA 13 supports sm_75 and newer
  EXPECT_GT(info.sm_count, 0);
  EXPECT_GT(info.max_threads_per_sm, 0);
  EXPECT_GT(info.shared_mem_per_block, 0u);
  EXPECT_GT(info.total_global_mem, 0u);
}

TEST(DeviceInfo, InvalidIndexThrows) {
  EXPECT_THROW(sfm::core::query_device(9999), sfm::core::CudaError);
}
