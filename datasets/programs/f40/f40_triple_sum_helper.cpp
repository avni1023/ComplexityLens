#include <vector>
#include <cstddef>

// Variant: helper-function split — inner linear scan is extracted to helper.
// The double outer loop calls the helper n^2 times; helper does n work -> O(n^3).
// Structurally distinct: inter-procedural decomposition.
static bool find_third(const std::vector<int>& v, std::size_t start, int needed) {
    for (std::size_t k = start; k < v.size(); ++k) {
        if (v[k] == needed) {
            return true;
        }
    }
    return false;
}

bool triple_sum_helper(const std::vector<int>& v, int target) {
    std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            int needed = target - v[i] - v[j];
            if (find_third(v, j + 1, needed)) {
                return true;
            }
        }
    }
    return false;
}
