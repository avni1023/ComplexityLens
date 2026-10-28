// Seed: recursive function making three self-calls on n-1.
// T(n) = 3T(n-1) + O(1) -> Theta(3^n) = OEXP.
long long three_calls_basic(int n) {
    if (n <= 0) {
        return 1;
    }
    long long a = three_calls_basic(n - 1);
    long long b = three_calls_basic(n - 1);
    long long c = three_calls_basic(n - 1);
    return a + b + c;
}
