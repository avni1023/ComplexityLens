#include <vector>

int accumulate_with_offset(const std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    int offset = 3;
    int bound = n - offset;
    int sum = 0;
    for (int i = 0; i < bound; ++i) {
        sum += arr[i];
    }
    return sum;
}
