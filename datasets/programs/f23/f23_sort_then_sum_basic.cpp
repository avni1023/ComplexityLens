#include <vector>
#include <algorithm>

int sort_and_sum(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    int total = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        total += v[i];
    }
    return total;
}
