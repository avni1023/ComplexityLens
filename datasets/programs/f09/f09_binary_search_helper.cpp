#include <vector>

int calculate_midpoint(int l, int r) {
    return l + (r - l) / 2;
}

bool binary_search_helper(const std::vector<int>& v, int key) {
    int left = 0;
    int right = static_cast<int>(v.size()) - 1;
    while (left <= right) {
        int mid = calculate_midpoint(left, right);
        if (v[mid] == key) {
            return true;
        }
        if (v[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return false;
}
