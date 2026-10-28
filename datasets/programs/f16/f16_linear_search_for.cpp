#include <vector>

int linear_search_for(const std::vector<int>& vec, int target) {
    int n = static_cast<int>(vec.size());
    for (int i = 0; i < n; ++i) {
        if (vec[i] == target) {
            return i;
        }
    }
    return -1;
}
