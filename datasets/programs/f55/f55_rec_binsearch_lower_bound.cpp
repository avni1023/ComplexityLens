#include <vector>

int rec_lower_bound(const std::vector<int>& arr, int lo, int hi, int val) {
    if (lo >= hi) {
        return lo;
    }
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] >= val) {
        return rec_lower_bound(arr, lo, mid, val);
    }
    return rec_lower_bound(arr, mid + 1, hi, val);
}
