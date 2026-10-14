#include <vector>

int sum_elements(const std::vector<int>& arr) {
    int total = 0;
    int n = static_cast<int>(arr.size());
    for (int i = 0; i < n; ++i) {
        total += arr[i];
    }
    return total;
}
