#include <vector>
#include <algorithm>
#include <cstddef>

bool contains_item(const std::vector<int>& container, int val) {
    return std::find(container.begin(), container.end(), val) != container.end();
}

int count_common_elements_helper(const std::vector<int>& first, const std::vector<int>& second) {
    int total = 0;
    for (std::size_t idx = 0; idx < first.size(); ++idx) {
        if (contains_item(second, first[idx])) {
            total++;
        }
    }
    return total;
}
