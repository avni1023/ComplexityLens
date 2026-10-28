// Seed: classic naive recursive Fibonacci.
// T(n) = T(n-1) + T(n-2) + O(1) -> OEXP (Theta(phi^n)).
long long naive_fib_basic(int n) {
    if (n <= 1) {
        return n;
    }
    return naive_fib_basic(n - 1) + naive_fib_basic(n - 2);
}
