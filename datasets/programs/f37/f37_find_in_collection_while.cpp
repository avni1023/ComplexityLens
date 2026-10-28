#include <vector>
#include <algorithm>

int count_common_elements_while(const std::vector<int>& a, const std::vector<int>& b) {
    int matches = 0;
    int i = 0;
    int n = static_cast<int>(a.size());
    while (i < n) {
        auto it = std::find(b.begin(), b.end(), a[i]);
        if (it != b.end()) {
            matches++;
        }
        ++i;
    }
    return matches;
}
