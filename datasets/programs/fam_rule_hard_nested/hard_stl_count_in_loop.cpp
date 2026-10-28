#include <vector>
#include <algorithm>

int count_cross_frequencies(const std::vector<int>& keys, const std::vector<int>& data) {
    int total_occurrences = 0;
    int n = static_cast<int>(keys.size());
    for (int i = 0; i < n; ++i) {
        total_occurrences += static_cast<int>(std::count(data.begin(), data.end(), keys[i]));
    }
    return total_occurrences;
}
