#include <vector>
#include <algorithm>
#include <cstddef>

int count_common_elements_for(const std::vector<int>& a, const std::vector<int>& b) {
    int common = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::find(b.begin(), b.end(), a[i]) != b.end()) {
            common++;
        }
    }
    return common;
}
