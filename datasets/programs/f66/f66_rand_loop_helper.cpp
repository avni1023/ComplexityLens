#include <cstdlib>

// Variant: helper-function split — helper draws random number and evaluates exit condition.
// Structurally distinct: random call encapsulated inside predicate helper.
// UNKNOWN / external_call.
static bool keep_rolling() {
    return (std::rand() % 10) != 0;
}

int rand_loop_helper() {
    int iterations = 0;
    while (keep_rolling()) {
        ++iterations;
    }
    return iterations;
}
