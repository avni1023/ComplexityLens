#include <vector>

int max_sliding_window_sum(const std::vector<int>& v, int k) {
    int n = static_cast<int>(v.size());
    if (k <= 0 || k > n) return 0;
    int current_sum = 0;
    for (int i = 0; i < k; i++) {
        current_sum += v[i];
    }
    int best_sum = current_sum;
    for (int i = k; i < n; i++) {
        current_sum += v[i] - v[i - k];
        if (current_sum > best_sum) {
            best_sum = current_sum;
        }
    }
    return best_sum;
}
