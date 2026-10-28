#include <cstdlib>

// Seed: while-loop exiting when std::rand() returns a multiple of 10.
// UNKNOWN / external_call: static analysis cannot predict or bound std::rand() outputs.
int rand_loop_basic() {
    int attempts = 0;
    while (std::rand() % 10 != 0) {
        ++attempts;
    }
    return attempts;
}
