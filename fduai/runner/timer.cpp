//===- fduai/runner/timer.cpp - High Resolution Timers --------------------===//
//
// Timers are used to measure the time taken for specific operations in the code.
// It provides llvm.func @timer_start() and llvm.func @timer_stop() functions.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

static std::chrono::_V2::system_clock::time_point timers[16];
static int n_timers = 0;

extern "C" void timer_start(void) {
    if (n_timers >= 16) {
        std::cerr << "Maximum number of timers reached." << std::endl;
        return;
    }
    auto timer = Clock::now();
    timers[n_timers++] = timer;
}

extern "C" void timer_stop(void) {
    if (n_timers <= 0) {
        std::cerr << "No timers to stop." << std::endl;
        return;
    }
    auto timer = Clock::now();
    auto duration = (timer - timers[n_timers - 1]);
    std::cerr << "[Timer " << n_timers - 1 << "] ";
    std::cerr << duration.count() << std::endl;
}
