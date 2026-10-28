#include <vector>
#include <cstddef>

// Seed: canonical triple nested for-loops checking for triple sum.
// T(n) = n * (n - 1) * (n - 2) / 6 * O(1) = Theta(n^3) -> ON3.
bool triple_sum_basic(const std::vector<int>& v, int target) {
    std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            for (std::size_t k = j + 1; k < n; ++k) {
                if (v[i] + v[j] + v[k] == target) {
                    return true;
                }
            }
        }
    }
    return false;
}
