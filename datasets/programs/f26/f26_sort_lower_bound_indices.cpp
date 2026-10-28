#include <vector>
#include <algorithm>

int sort_and_query_bounds(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    int matched_positions = 0;
    int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
        auto it = std::lower_bound(v.begin(), v.end(), v[i]);
        matched_positions += static_cast<int>(it - v.begin());
    }
    return matched_positions;
}
