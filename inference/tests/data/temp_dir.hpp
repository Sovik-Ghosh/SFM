#pragma once
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace sfm::data::test {

// A per-test temporary directory, removed afterwards.
class TempDirTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = std::filesystem::temp_directory_path() /
           (std::string("sfm_") + info->test_suite_name() + "_" + info->name());
    std::filesystem::remove_all(dir_);
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  std::filesystem::path write(const std::string& name, const std::string& bytes) const {
    const auto p = dir_ / name;
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << bytes;
    return p;
  }

  // Binary PNM: magic "P6" (RGB) or "P5" (gray), maxval 255.
  std::filesystem::path write_pnm(const std::string& name, const char* magic, int w, int h,
                                  const std::string& pixels) const {
    return write(name, std::string(magic) + "\n" + std::to_string(w) + " " + std::to_string(h) +
                           "\n255\n" + pixels);
  }

  std::filesystem::path dir_;
};

}  // namespace sfm::data::test
