#include <vector>

int find_min_index(const std::vector<int>& arr, int start, int end) {
    int min_idx = start;
    for (int k = start + 1; k < end; ++k) {
        if (arr[k] < arr[min_idx]) {
            min_idx = k;
        }
    }
    return min_idx;
}

void selection_sort_with_helper(std::vector<int>& data) {
    int n = static_cast<int>(data.size());
    for (int i = 0; i < n; ++i) {
        int m = find_min_index(data, i, n);
        int tmp = data[i];
        data[i] = data[m];
        data[m] = tmp;
    }
}
