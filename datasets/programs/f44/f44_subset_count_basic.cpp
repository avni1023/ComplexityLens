#include <vector>

// Seed: classic recursive subset count (two recursive branches on n-1).
// T(n) = 2T(n-1) + O(1) -> Theta(2^n) = OEXP.
long long subset_count_basic(const std::vector<int>& v, int n, int target) {
    if (n <= 0) {
        return (target == 0) ? 1 : 0;
    }
    long long exclude_count = subset_count_basic(v, n - 1, target);
    long long include_count = subset_count_basic(v, n - 1, target - v[n - 1]);
    return exclude_count + include_count;
}
