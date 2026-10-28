// Seed: Collatz step counter using a for-style idiom (no fixed bound).
// UNKNOWN/termination: the Collatz conjecture is unproven; static analysis
// cannot determine a worst-case step count for arbitrary positive n.
long long collatz_steps_basic(long long n) {
    if (n <= 0) {
        return 0;
    }
    long long steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        ++steps;
    }
    return steps;
}
