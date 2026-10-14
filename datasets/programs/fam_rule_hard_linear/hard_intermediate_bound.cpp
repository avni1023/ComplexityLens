#include <vector>

int sum_half_range(const std::vector<int>& vec) {
    int n = static_cast<int>(vec.size());
    int limit = n / 2;
    int total = 0;
    for (int i = 0; i < limit; ++i) {
        total += vec[i];
    }
    return total;
}
