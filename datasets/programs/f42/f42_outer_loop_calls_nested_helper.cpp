// Seed: for-loop calling a helper with two nested for-loops.
// Outer: O(n); helper: O(n^2); total: O(n^3).
static long long nested_square_work(int n) {
    long long acc = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            acc += i * n + j;
        }
    }
    return acc;
}

long long outer_loop_calls_nested_helper(int n) {
    long long total = 0;
    for (int k = 0; k < n; ++k) {
        total += nested_square_work(n);
    }
    return total;
}
