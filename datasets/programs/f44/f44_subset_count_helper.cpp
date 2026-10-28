#include <vector>

// Variant: helper-function split — base case and choice decision are in helpers.
// Structurally distinct: base case check separated into helper predicate.
// T(n) = 2T(n-1) + O(1) -> OEXP.
static bool is_terminal(int n) {
    return n <= 0;
}

static long long terminal_value(int target) {
    return (target == 0) ? 1 : 0;
}

long long subset_count_helper(const std::vector<int>& v, int n, int target) {
    if (is_terminal(n)) {
        return terminal_value(target);
    }
    long long without_elem = subset_count_helper(v, n - 1, target);
    long long with_elem = subset_count_helper(v, n - 1, target - v[n - 1]);
    return without_elem + with_elem;
}
