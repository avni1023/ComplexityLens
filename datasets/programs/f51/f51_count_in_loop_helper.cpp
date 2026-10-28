#include <vector>
#include <algorithm>
#include <cstddef>

// Variant: helper-function split — wrapper helper encapsulates the std::count call.
// Structurally distinct: inter-procedural call to wrapper helper containing std::count.
// n * O(n) = O(n^2).
static int count_occurrences(const std::vector<int>& v, int target) {
    return static_cast<int>(std::count(v.begin(), v.end(), target));
}

long long count_in_loop_helper(const std::vector<int>& v) {
    long long sum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        sum += count_occurrences(v, v[i]);
    }
    return sum;
}
