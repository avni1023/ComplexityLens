#include <vector>

std::vector<int> all_window_sums(const std::vector<int>& arr, int k) {
    int n = static_cast<int>(arr.size());
    if (k <= 0 || k > n) return {};
    std::vector<int> result;
    int w_sum = 0;
    for (int i = 0; i < k; i++) {
        w_sum += arr[i];
    }
    result.push_back(w_sum);
    for (int r = k; r < n; r++) {
        w_sum += arr[r] - arr[r - k];
        result.push_back(w_sum);
    }
    return result;
}
