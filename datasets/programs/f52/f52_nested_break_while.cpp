#include <vector>

int nested_while_sentinel_break(const std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    int accumulated = 0;
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n) {
            accumulated += (i * n + j);
            if (arr[j] < 0) {
                break;
            }
            ++j;
        }
        ++i;
    }
    return accumulated;
}
