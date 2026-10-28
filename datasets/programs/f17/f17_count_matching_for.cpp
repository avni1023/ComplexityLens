#include <vector>

int count_even_numbers(const std::vector<int>& vec) {
    int count = 0;
    int n = static_cast<int>(vec.size());
    for (int i = 0; i < n; ++i) {
        if (vec[i] % 2 == 0) {
            count++;
        }
    }
    return count;
}
