#include <vector>

int helper_scan(const std::vector<int>& vec, int limit) {
    int acc = 0;
    for (int j = 0; j < limit; ++j) {
        acc += vec[j];
    }
    return acc;
}

int outer_nested_caller(const std::vector<int>& vec) {
    int n = static_cast<int>(vec.size());
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += helper_scan(vec, n);
    }
    return total;
}
