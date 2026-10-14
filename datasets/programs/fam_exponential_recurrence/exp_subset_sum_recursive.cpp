#include <vector>

bool subset_sum(const std::vector<int>& arr, int n, int target) {
    if (target == 0) return true;
    if (n == 0) return false;
    return subset_sum(arr, n - 1, target) ||
           subset_sum(arr, n - 1, target - arr[n - 1]);
}
