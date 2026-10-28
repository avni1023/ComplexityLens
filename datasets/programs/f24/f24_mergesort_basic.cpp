#include <vector>
#include <algorithm>

static void merge_ranges(std::vector<int>& arr, int lo, int mid, int hi) {
    std::vector<int> tmp(arr.begin() + lo, arr.begin() + mid + 1);
    int i = 0;
    int j = mid + 1;
    int k = lo;
    while (i < static_cast<int>(tmp.size()) && j <= hi) {
        if (tmp[i] <= arr[j]) {
            arr[k++] = tmp[i++];
        } else {
            arr[k++] = arr[j++];
        }
    }
    while (i < static_cast<int>(tmp.size())) {
        arr[k++] = tmp[i++];
    }
}

void mergesort_basic(std::vector<int>& arr, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int mid = lo + (hi - lo) / 2;
    mergesort_basic(arr, lo, mid);
    mergesort_basic(arr, mid + 1, hi);
    merge_ranges(arr, lo, mid, hi);
}
