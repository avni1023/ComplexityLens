#include <vector>

int dac_find_max(const std::vector<int>& vec, int lo, int hi) {
    if (lo == hi) {
        return vec[lo];
    }
    int mid = lo + (hi - lo) / 2;
    int m1 = dac_find_max(vec, lo, mid);
    int m2 = dac_find_max(vec, mid + 1, hi);
    return (m1 > m2) ? m1 : m2;
}

int divide_conquer_max_element(const std::vector<int>& items) {
    if (items.empty()) {
        return 0;
    }
    return dac_find_max(items, 0, static_cast<int>(items.size()) - 1);
}
