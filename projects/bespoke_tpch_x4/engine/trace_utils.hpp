#pragma once

#ifdef TRACE
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>

namespace trace_utils {
inline uint64_t get_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

class ScopedTimer {
public:
    ScopedTimer(const char* name, void (*record)(const char*, uint64_t))
        : name_(name), record_(record), start_ns_(get_time_ns()) {}

    ~ScopedTimer() { record_(name_, get_time_ns() - start_ns_); }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* name_;
    void (*record_)(const char*, uint64_t);
    uint64_t start_ns_;
};

inline std::ostream& trace_output() {
    static std::ofstream out("tracing_output.log", std::ios::app);
    return out;
}

class TraceOutputGuard {
public:
    TraceOutputGuard() : old_buf_(std::cout.rdbuf(trace_output().rdbuf())) {}

    ~TraceOutputGuard() {
        trace_output().flush();
        std::cout.rdbuf(old_buf_);
    }

    TraceOutputGuard(const TraceOutputGuard&) = delete;
    TraceOutputGuard& operator=(const TraceOutputGuard&) = delete;

private:
    std::streambuf* old_buf_;
};
}  // namespace trace_utils
#endif