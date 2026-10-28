#include <vector>

int binary_search_half_open(const std::vector<int>& arr, int target) {
    int low = 0;
    int high = static_cast<int>(arr.size());
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] >= target) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    if (low < static_cast<int>(arr.size()) && arr[low] == target) {
        return low;
    }
    return -1;
}
