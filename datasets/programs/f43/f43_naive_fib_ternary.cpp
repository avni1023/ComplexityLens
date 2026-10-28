// Variant: ternary-expression style base case — same two recursive calls,
// different syntactic form for the base-case guard.
long long naive_fib_ternary(int n) {
    return (n <= 1) ? static_cast<long long>(n)
                    : naive_fib_ternary(n - 1) + naive_fib_ternary(n - 2);
}
