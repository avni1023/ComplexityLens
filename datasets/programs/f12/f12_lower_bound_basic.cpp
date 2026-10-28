#include <vector>
#include <algorithm>

int find_lower_bound_index(const std::vector<int>& vec, int val) {
    auto it = std::lower_bound(vec.begin(), vec.end(), val);
    if (it != vec.end()) {
        return static_cast<int>(it - vec.begin());
    }
    return -1;
}
