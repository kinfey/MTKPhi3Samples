#pragma once

#include <chrono>

class Timer {
public:
    Timer() {};

    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_).count();
    }

    double reset() {
        const auto now = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        start_ = now;
        return elapsed;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};