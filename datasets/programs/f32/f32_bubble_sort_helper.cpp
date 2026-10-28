#include <vector>

bool bubble_pass(std::vector<int>& vec, int limit) {
    bool swapped = false;
    for (int j = 0; j < limit; ++j) {
        if (vec[j] > vec[j + 1]) {
            int t = vec[j];
            vec[j] = vec[j + 1];
            vec[j + 1] = t;
            swapped = true;
        }
    }
    return swapped;
}

void bubble_sort_helper(std::vector<int>& vec) {
    int n = static_cast<int>(vec.size());
    for (int i = 0; i < n - 1; ++i) {
        bubble_pass(vec, n - i - 1);
    }
}
