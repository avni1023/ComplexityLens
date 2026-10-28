#include <vector>
#include <cstddef>

bool matches_target(int value, int target) {
    return value == target;
}

int linear_search_helper(const std::vector<int>& v, int target) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (matches_target(v[i], target)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}
