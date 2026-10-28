#include <vector>

int two_loops_sum_max(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += v[i];
    }
    int max_val = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i] > max_val) {
            max_val = v[i];
        }
    }
    return total + max_val;
}
