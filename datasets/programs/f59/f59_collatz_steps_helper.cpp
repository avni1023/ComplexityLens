// Variant: helper-function split — the single Collatz step is extracted into a helper.
// The main function drives the loop; the helper performs the parity check and update.
// UNKNOWN/termination: termination of the sequence for all positive integers is
// the 3n+1 (Collatz) conjecture — unproven; static analysis must abstain.
static long long collatz_step(long long n) {
    return (n % 2 == 0) ? n / 2 : 3 * n + 1;
}

long long collatz_steps_helper(long long n) {
    if (n <= 0) {
        return 0;
    }
    long long steps = 0;
    while (n != 1) {
        n = collatz_step(n);
        ++steps;
    }
    return steps;
}
