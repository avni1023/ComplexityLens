#include <vector>

int binary_search_standard(const std::vector<int>& vec, int target) {
    int low = 0;
    int high = static_cast<int>(vec.size()) - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (vec[mid] == target) {
            return mid;
        } else if (vec[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}
