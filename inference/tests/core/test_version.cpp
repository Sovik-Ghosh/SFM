#include <gtest/gtest.h>

#include <string>

#include "sfm/core/version.hpp"

TEST(Version, LibraryMatchesHeaders) {
  EXPECT_STREQ(sfm::core::version_string(), sfm::core::kVersionString);
}

TEST(Version, StringIsMajorMinorPatch) {
  const std::string expected = std::to_string(sfm::core::kVersionMajor) + "." +
                               std::to_string(sfm::core::kVersionMinor) + "." +
                               std::to_string(sfm::core::kVersionPatch);
  EXPECT_EQ(sfm::core::version_string(), expected);
}
