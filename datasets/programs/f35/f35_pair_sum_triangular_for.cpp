#include <vector>

int count_matching_pairs(const std::vector<int>& v, int target) {
    int matches = 0;
    int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (v[i] + v[j] == target) {
                matches++;
            }
        }
    }
    return matches;
}
