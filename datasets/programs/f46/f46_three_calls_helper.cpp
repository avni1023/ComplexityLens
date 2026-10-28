// Variant: helper-function split — combine step is in a helper function.
// Structurally distinct: combines three branches through a dedicated adder helper.
// T(n) = 3T(n-1) + O(1) -> OEXP.
static long long add_three(long long x, long long y, long long z) {
    return x + y + z;
}

long long three_calls_helper(int n) {
    if (n <= 0) {
        return 1;
    }
    return add_three(
        three_calls_helper(n - 1),
        three_calls_helper(n - 1),
        three_calls_helper(n - 1)
    );
}
