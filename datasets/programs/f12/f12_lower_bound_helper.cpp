#include <vector>
#include <algorithm>

std::vector<int>::const_iterator search_bound(const std::vector<int>& v, int x) {
    return std::lower_bound(v.begin(), v.end(), x);
}

int lower_bound_with_helper(const std::vector<int>& v, int x) {
    auto it = search_bound(v, x);
    if (it != v.end()) {
        return *it;
    }
    return -1;
}
