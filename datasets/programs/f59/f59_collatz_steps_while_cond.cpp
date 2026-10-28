// Variant: the Collatz condition is embedded directly in the while-loop condition
// via a helper predicate, giving a different structural form (condition-in-while style).
// UNKNOWN/termination: same open-conjecture bound applies regardless of loop syntax.
static bool collatz_not_done(long long n) {
    return n != 1;
}

long long collatz_steps_while_cond(long long n) {
    if (n <= 0) {
        return 0;
    }
    long long steps = 0;
    while (collatz_not_done(n)) {
        n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
        ++steps;
    }
    return steps;
}
