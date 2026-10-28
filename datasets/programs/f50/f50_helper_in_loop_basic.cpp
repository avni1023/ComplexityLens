// Seed: outer for-loop of size n calls helper that runs a for-loop of size n.
// Inter-procedural composition: n * n = O(n^2).
static long long inner_linear_helper(int n) {
    long long s = 0;
    for (int j = 0; j < n; ++j) {
        s += j;
    }
    return s;
}

long long helper_in_loop_basic(int n) {
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        total += inner_linear_helper(n);
    }
    return total;
}
