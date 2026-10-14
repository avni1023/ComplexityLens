#include <vector>

bool has_duplicates(const std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (arr[i] == arr[j]) {
                return true;
            }
        }
    }
    return false;
}
