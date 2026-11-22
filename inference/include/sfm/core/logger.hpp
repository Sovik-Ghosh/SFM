#pragma once
// Structured logger: one line per event, `key=value` fields, e.g.
//   2026-11-15T10:00:00.123Z level=info event=dataset.loaded images=36 masks=true
// Values containing spaces, quotes or '=' are double-quoted. Thread-safe. No global logger:
// the Pipeline owns one and passes it down.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace sfm::core {

enum class LogLevel : std::uint8_t { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

// "debug" | "info" | "warn" | "error"; throws ConfigError otherwise.
LogLevel parse_log_level(std::string_view name);
std::string_view to_string(LogLevel level);

struct LogField {
  LogField(std::string_view k, std::string_view v) : key(k), value(v) {}
  LogField(std::string_view k, const char* v) : key(k), value(v) {}
  LogField(std::string_view k, bool v) : key(k), value(v ? "true" : "false") {}
  template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  LogField(std::string_view k, T v) : key(k), value(format_number(v)) {}

  std::string key;
  std::string value;

 private:
  static std::string format_number(double v);
  static std::string format_number(long long v) { return std::to_string(v); }
  static std::string format_number(unsigned long long v) { return std::to_string(v); }
  template <class T>
  static std::string format_number(T v) {
    if constexpr (std::is_floating_point_v<T>) {
      return format_number(static_cast<double>(v));
    } else if constexpr (std::is_signed_v<T>) {
      return format_number(static_cast<long long>(v));
    } else {
      return format_number(static_cast<unsigned long long>(v));
    }
  }
};

class Logger {
 public:
  // `sink` must outlive the logger. timestamps=false gives deterministic lines (tests).
  explicit Logger(LogLevel min_level = LogLevel::kInfo, std::ostream* sink = nullptr,
                  bool timestamps = true);

  void set_level(LogLevel level) noexcept { min_level_ = level; }
  LogLevel level() const noexcept { return min_level_; }
  // Also append every line to `path` (replacing a file set by an earlier call). Throws IoError if
  // the file cannot be opened.
  void add_file(const std::filesystem::path& path);

  void log(LogLevel level, std::string_view event, std::initializer_list<LogField> fields = {});
  void debug(std::string_view e, std::initializer_list<LogField> f = {}) {
    log(LogLevel::kDebug, e, f);
  }
  void info(std::string_view e, std::initializer_list<LogField> f = {}) {
    log(LogLevel::kInfo, e, f);
  }
  void warn(std::string_view e, std::initializer_list<LogField> f = {}) {
    log(LogLevel::kWarn, e, f);
  }
  void error(std::string_view e, std::initializer_list<LogField> f = {}) {
    log(LogLevel::kError, e, f);
  }

 private:
  LogLevel min_level_;
  std::ostream* sink_;
  bool timestamps_;
  std::ofstream file_;
  std::mutex mutex_;
};

}  // namespace sfm::core
