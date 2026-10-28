#include <vector>

int find_with_early_exit(const std::vector<int>& arr, int target) {
    int n = static_cast<int>(arr.size());
    int found_index = -1;
    for (int i = 0; i < n; ++i) {
        if (arr[i] == target) {
            found_index = i;
            break;
        }
    }
    return found_index;
}
