#include <vector>
#include <algorithm>

bool find_pair_sum_sorted(std::vector<int>& arr, int target) {
    std::sort(arr.begin(), arr.end());
    int left = 0;
    int right = static_cast<int>(arr.size()) - 1;
    while (left < right) {
        int s = arr[left] + arr[right];
        if (s == target) {
            return true;
        }
        if (s < target) {
            ++left;
        } else {
            --right;
        }
    }
    return false;
}
