#include <vector>

long long add_to_sum(long long current, int value) {
    return current + value;
}

long long sum_vector_helper(const std::vector<int>& vec) {
    long long total = 0;
    int n = static_cast<int>(vec.size());
    for (int i = 0; i < n; ++i) {
        total = add_to_sum(total, vec[i]);
    }
    return total;
}
