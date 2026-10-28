#include <vector>
#include <algorithm>

int sort_lower_bound_offsets(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
    int offset_hits = 0;
    int n = static_cast<int>(vec.size());
    int i = 0;
    while (i < n) {
        auto it = std::lower_bound(vec.begin(), vec.end(), vec[i] - 1);
        offset_hits += static_cast<int>(vec.end() - it);
        ++i;
    }
    return offset_hits;
}
