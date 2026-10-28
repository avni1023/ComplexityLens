#include <vector>
#include <algorithm>

int count_unique_after_sort(std::vector<int>& v) {
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    int unique_count = 1;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] != v[i - 1]) {
            unique_count++;
        }
    }
    return unique_count;
}
