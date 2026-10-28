#include <vector>
#include <algorithm>
#include <cstddef>

// Seed: for-loop over vector v calling std::count(v.begin(), v.end(), v[i]).
// Outer loop n times * std::count taking n work = O(n^2).
long long count_in_loop_basic(const std::vector<int>& v) {
    long long total_occurrences = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        total_occurrences += std::count(v.begin(), v.end(), v[i]);
    }
    return total_occurrences;
}
