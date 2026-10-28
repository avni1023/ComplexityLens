#include <vector>

int recursive_data_split(const std::vector<int>& arr, int lo, int hi) {
    if (lo >= hi - 1) {
        return 0;
    }
    int split_idx = arr[lo];
    if (split_idx <= lo || split_idx >= hi) {
        split_idx = lo + 1;
    }
    return 1 + recursive_data_split(arr, lo, split_idx) + recursive_data_split(arr, split_idx, hi);
}
