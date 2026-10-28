#include <vector>
#include <cstddef>

int find_max_element(const std::vector<int>& v) {
    if (v.empty()) {
        return -1;
    }
    int max_val = v[0];
    for (std::size_t i = 1; i < v.size(); i++) {
        if (v[i] > max_val) {
            max_val = v[i];
        }
    }
    return max_val;
}
