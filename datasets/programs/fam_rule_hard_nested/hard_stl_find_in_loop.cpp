#include <vector>
#include <algorithm>

int count_matching_elements(const std::vector<int>& a, const std::vector<int>& b) {
    int matches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::find(b.begin(), b.end(), a[i]) != b.end()) {
            matches++;
        }
    }
    return matches;
}
