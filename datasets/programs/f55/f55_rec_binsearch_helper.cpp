#include <vector>

int binary_search_worker(const int* data, int left, int right, int key) {
    if (left > right) {
        return -1;
    }
    int mid = left + (right - left) / 2;
    if (data[mid] == key) {
        return mid;
    }
    if (data[mid] < key) {
        return binary_search_worker(data, mid + 1, right, key);
    }
    return binary_search_worker(data, left, mid - 1, key);
}

int recursive_search_driver(const std::vector<int>& vec, int target) {
    if (vec.empty()) {
        return -1;
    }
    return binary_search_worker(vec.data(), 0, static_cast<int>(vec.size()) - 1, target);
}
