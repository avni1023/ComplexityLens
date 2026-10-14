#include <vector>
#include <algorithm>

int search_sorted(const std::vector<int>& vec, int target) {
    auto it = std::lower_bound(vec.begin(), vec.end(), target);
    if (it != vec.end()) {
        return *it;
    }
    return -1;
}
