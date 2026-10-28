#include <vector>

void linear_combine(std::vector<int>& arr, int start, int mid, int end) {
    std::vector<int> buf;
    int p1 = start, p2 = mid;
    while (p1 < mid && p2 < end) {
        if (arr[p1] < arr[p2]) buf.push_back(arr[p1++]);
        else buf.push_back(arr[p2++]);
    }
    while (p1 < mid) buf.push_back(arr[p1++]);
    while (p2 < end) buf.push_back(arr[p2++]);
    for (int k = 0; k < static_cast<int>(buf.size()); ++k) {
        arr[start + k] = buf[k];
    }
}

void sort_range(std::vector<int>& arr, int start, int end) {
    if (end - start > 1) {
        int mid = start + (end - start) / 2;
        sort_range(arr, start, mid);
        sort_range(arr, mid, end);
        linear_combine(arr, start, mid, end);
    }
}
