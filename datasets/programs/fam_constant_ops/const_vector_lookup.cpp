#include <vector>

int lookup_element(const std::vector<int>& arr) {
    if (arr.empty()) {
        return -1;
    }
    return arr[0] + arr[arr.size() - 1];
}
