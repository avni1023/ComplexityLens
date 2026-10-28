#include <cstdlib>

// Variant: for-loop condition checking roll of a 6-sided die via std::rand().
// Structurally distinct: for-loop syntax with accumulated random values.
// UNKNOWN / external_call.
int rand_loop_for() {
    int total = 0;
    for (; std::rand() % 6 != 0; ) {
        total += 1;
    }
    return total;
}
