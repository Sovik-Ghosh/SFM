#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#include "sfm/core/error.hpp"
#include "sfm/core/logger.hpp"

using sfm::core::Logger;
using sfm::core::LogLevel;

TEST(Logger, FormatsKeyValueFields) {
  std::ostringstream out;
  Logger log(LogLevel::kDebug, &out, /*timestamps=*/false);
  log.info("dataset.loaded",
           {{"images", 36}, {"masks", true}, {"root", "data/bunny"}, {"fx", 1275.13}});
  EXPECT_EQ(out.str(),
            "level=info event=dataset.loaded images=36 masks=true root=data/bunny fx=1275.13\n");
}

TEST(Logger, QuotesValuesThatNeedIt) {
  std::ostringstream out;
  Logger log(LogLevel::kInfo, &out, false);
  log.warn("x", {{"msg", "two words"}, {"eq", "a=b"}, {"q", "say \"hi\""}, {"empty", ""}});
  EXPECT_EQ(out.str(),
            "level=warn event=x msg=\"two words\" eq=\"a=b\" q=\"say \\\"hi\\\"\" empty=\"\"\n");
}

TEST(Logger, FiltersBelowMinimumLevel) {
  std::ostringstream out;
  Logger log(LogLevel::kWarn, &out, false);
  log.debug("a");
  log.info("b");
  log.warn("c");
  log.error("d");
  EXPECT_EQ(out.str(), "level=warn event=c\nlevel=error event=d\n");
}

TEST(Logger, TimestampIsIso8601Utc) {
  std::ostringstream out;
  Logger log(LogLevel::kInfo, &out, true);
  log.info("t");
  const std::regex re(R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z level=info event=t\n$)");
  EXPECT_TRUE(std::regex_match(out.str(), re)) << out.str();
}

TEST(Logger, AlsoWritesToFile) {
  const auto path = std::filesystem::temp_directory_path() / "sfm_test_logger.log";
  std::filesystem::remove(path);
  {
    std::ostringstream out;
    Logger log(LogLevel::kInfo, &out, false);
    log.add_file(path);
    log.info("to.file", {{"n", 1}});
  }
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  EXPECT_EQ(line, "level=info event=to.file n=1");
  std::filesystem::remove(path);
}

TEST(Logger, BadFileThrowsIoError) {
  Logger log;
  EXPECT_THROW(log.add_file("/nonexistent-dir/x.log"), sfm::IoError);
}

TEST(Logger, SecondFileReplacesTheFirst) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto a = dir / "sfm_logger_a.log";
  const auto b = dir / "sfm_logger_b.log";
  std::filesystem::remove(a);
  std::filesystem::remove(b);
  {
    sfm::core::Logger log(sfm::core::LogLevel::kInfo, nullptr, false);
    log.add_file(a);
    log.info("first");
    log.add_file(b);  // e.g. one logger, two pipeline runs
    log.info("second");
  }
  const auto read = [](const std::filesystem::path& p) {
    std::ifstream in(p);
    std::string all;
    for (std::string line; std::getline(in, line);) all += line + "\n";
    return all;
  };
  EXPECT_EQ(read(a), "level=info event=first\n");
  EXPECT_EQ(read(b), "level=info event=second\n");
  std::filesystem::remove(a);
  std::filesystem::remove(b);
}

TEST(LogLevel, ParsesAndRejects) {
  EXPECT_EQ(sfm::core::parse_log_level("debug"), LogLevel::kDebug);
  EXPECT_EQ(sfm::core::parse_log_level("error"), LogLevel::kError);
  EXPECT_THROW(sfm::core::parse_log_level("INFO"), sfm::ConfigError);  // case-sensitive
  EXPECT_THROW(sfm::core::parse_log_level(""), sfm::ConfigError);
  EXPECT_EQ(sfm::core::to_string(LogLevel::kWarn), "warn");
}
