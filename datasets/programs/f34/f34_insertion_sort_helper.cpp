#include <vector>

void insert_into_prefix(std::vector<int>& arr, int index) {
    int val = arr[index];
    int pos = index - 1;
    while (pos >= 0 && arr[pos] > val) {
        arr[pos + 1] = arr[pos];
        --pos;
    }
    arr[pos + 1] = val;
}

void insertion_sort_modular(std::vector<int>& data) {
    int n = static_cast<int>(data.size());
    for (int i = 1; i < n; ++i) {
        insert_into_prefix(data, i);
    }
}
