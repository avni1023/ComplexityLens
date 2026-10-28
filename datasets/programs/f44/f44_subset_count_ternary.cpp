#include <vector>

// Variant: ternary-expression base case and single-expression sum.
// Structurally distinct: no explicit if statements; returns compound expression.
// T(n) = 2T(n-1) + O(1) -> OEXP.
long long subset_count_ternary(const std::vector<int>& v, int n, int target) {
    return (n <= 0)
        ? (target == 0 ? 1LL : 0LL)
        : (subset_count_ternary(v, n - 1, target) +
           subset_count_ternary(v, n - 1, target - v[n - 1]));
}
