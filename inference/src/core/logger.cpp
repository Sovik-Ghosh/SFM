#include "sfm/core/logger.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>

#include "sfm/core/error.hpp"

namespace sfm::core {

namespace {

bool needs_quotes(std::string_view v) {
  if (v.empty()) return true;
  for (const char c : v) {
    if (c == ' ' || c == '"' || c == '=' || c == '\t' || c == '\n') return true;
  }
  return false;
}

std::string quote(std::string_view v) {
  std::string out = "\"";
  for (const char c : v) {
    if (c == '"' || c == '\\') out += '\\';
    out += (c == '\n') ? ' ' : c;
  }
  return out + "\"";
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char date[32];
  std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &tm);
  char frac[16];  // ".mmmZ"; sized for any int so the compiler can prove no truncation
  std::snprintf(frac, sizeof(frac), ".%03dZ", static_cast<int>(ms.count()));
  return std::string(date) + frac;
}

}  // namespace

LogLevel parse_log_level(std::string_view name) {
  if (name == "debug") return LogLevel::kDebug;
  if (name == "info") return LogLevel::kInfo;
  if (name == "warn") return LogLevel::kWarn;
  if (name == "error") return LogLevel::kError;
  throw ConfigError("log level must be debug|info|warn|error, got \"" + std::string(name) + "\"");
}

std::string_view to_string(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kError:
      return "error";
  }
  return "unknown";
}

std::string LogField::format_number(double v) {
  std::ostringstream os;
  os.precision(6);
  os << v;
  return os.str();
}

Logger::Logger(LogLevel min_level, std::ostream* sink, bool timestamps)
    : min_level_(min_level), sink_(sink != nullptr ? sink : &std::cerr), timestamps_(timestamps) {}

void Logger::add_file(const std::filesystem::path& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (file_.is_open()) file_.close();  // a second call replaces the file (open() on an open
  file_.clear();                       // stream fails and would leave failbit set)
  file_.open(path, std::ios::app);
  if (!file_) throw IoError("cannot open log file " + path.string());
}

void Logger::log(LogLevel level, std::string_view event, std::initializer_list<LogField> fields) {
  if (level < min_level_) return;
  std::string line;
  if (timestamps_) line = utc_timestamp() + " ";
  line += "level=" + std::string(to_string(level)) + " event=" + std::string(event);
  for (const auto& f : fields) {
    line += " " + f.key + "=" + (needs_quotes(f.value) ? quote(f.value) : f.value);
  }
  line += '\n';

  std::lock_guard<std::mutex> lock(mutex_);
  *sink_ << line << std::flush;
  if (file_.is_open()) file_ << line << std::flush;
}

}  // namespace sfm::core
