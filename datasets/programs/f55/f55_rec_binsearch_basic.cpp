#include <vector>

int rec_binary_search(const std::vector<int>& v, int low, int high, int target) {
    if (low > high) {
        return -1;
    }
    int mid = low + (high - low) / 2;
    if (v[mid] == target) {
        return mid;
    }
    if (v[mid] > target) {
        return rec_binary_search(v, low, mid - 1, target);
    }
    return rec_binary_search(v, mid + 1, high, target);
}
