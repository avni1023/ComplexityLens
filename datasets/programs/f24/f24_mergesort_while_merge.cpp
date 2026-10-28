#include <vector>

// Variant: merge step uses while-loops with explicit index counters instead of
// a copied temporary; structurally distinct (three-way while merge in-place buffer).
static void merge_with_while(std::vector<int>& arr, int lo, int mid, int hi) {
    std::vector<int> left(arr.begin() + lo, arr.begin() + mid + 1);
    std::vector<int> right(arr.begin() + mid + 1, arr.begin() + hi + 1);
    int i = 0;
    int j = 0;
    int k = lo;
    while (i < static_cast<int>(left.size()) && j < static_cast<int>(right.size())) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    while (i < static_cast<int>(left.size())) {
        arr[k++] = left[i++];
    }
    while (j < static_cast<int>(right.size())) {
        arr[k++] = right[j++];
    }
}

void mergesort_while_merge(std::vector<int>& arr, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int mid = lo + (hi - lo) / 2;
    mergesort_while_merge(arr, lo, mid);
    mergesort_while_merge(arr, mid + 1, hi);
    merge_with_while(arr, lo, mid, hi);
}
