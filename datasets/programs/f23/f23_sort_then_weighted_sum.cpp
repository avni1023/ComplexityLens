#include <vector>
#include <algorithm>

long long sort_weighted_sum(std::vector<int>& vec) {
    std::sort(vec.begin(), vec.end());
    long long weighted = 0;
    int n = static_cast<int>(vec.size());
    for (int i = 0; i < n; ++i) {
        weighted += static_cast<long long>(i) * vec[i];
    }
    return weighted;
}
