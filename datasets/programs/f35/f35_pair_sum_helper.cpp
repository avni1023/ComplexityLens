#include <vector>

int count_pairs_for_elem(const std::vector<int>& data, int i, int target) {
    int matches = 0;
    int n = static_cast<int>(data.size());
    for (int j = i + 1; j < n; ++j) {
        if (data[i] + data[j] == target) {
            matches++;
        }
    }
    return matches;
}

int count_total_pairs_helper(const std::vector<int>& items, int target) {
    int total = 0;
    int n = static_cast<int>(items.size());
    for (int i = 0; i < n; ++i) {
        total += count_pairs_for_elem(items, i, target);
    }
    return total;
}
