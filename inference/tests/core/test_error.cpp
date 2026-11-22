#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>

#if SFM_WITH_CUDA
#include "sfm/core/cuda_check.h"
#endif
#include "sfm/core/error.hpp"

TEST(Error, HierarchyLetsAppsCatchOneType) {
  static_assert(std::is_base_of_v<std::runtime_error, sfm::Error>);
  static_assert(std::is_base_of_v<sfm::Error, sfm::ConfigError>);
  static_assert(std::is_base_of_v<sfm::Error, sfm::IoError>);
  static_assert(std::is_base_of_v<sfm::Error, sfm::ArenaError>);
#if SFM_WITH_CUDA
  static_assert(std::is_base_of_v<sfm::Error, sfm::core::CudaError>);
#endif
  SUCCEED();
}

TEST(Error, MessageIsKept) {
  try {
    throw sfm::ConfigError("dataset.root: missing");
  } catch (const sfm::Error& e) {
    EXPECT_STREQ(e.what(), "dataset.root: missing");
  }
}
