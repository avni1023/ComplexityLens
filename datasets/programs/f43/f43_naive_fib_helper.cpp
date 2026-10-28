// Variant: helper-function split — the base-case check is in a helper predicate;
// the recursive work is in the main function. Structurally distinct from both
// seed (if-return) and ternary variant.
static bool is_base_case(int n) {
    return n <= 1;
}

long long naive_fib_helper(int n) {
    if (is_base_case(n)) {
        return static_cast<long long>(n);
    }
    return naive_fib_helper(n - 1) + naive_fib_helper(n - 2);
}
