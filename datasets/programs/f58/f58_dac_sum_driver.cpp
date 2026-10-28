#include <vector>

int dac_sum_worker(const int* arr, int left, int right) {
    int count = right - left + 1;
    if (count <= 0) {
        return 0;
    }
    if (count == 1) {
        return arr[left];
    }
    int mid = left + (right - left) / 2;
    return dac_sum_worker(arr, left, mid) + dac_sum_worker(arr, mid + 1, right);
}

int divide_conquer_sum_driver(const std::vector<int>& data) {
    if (data.empty()) {
        return 0;
    }
    return dac_sum_worker(data.data(), 0, static_cast<int>(data.size()) - 1);
}
