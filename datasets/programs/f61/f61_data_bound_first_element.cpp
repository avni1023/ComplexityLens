#include <vector>

int loop_bound_by_first_element(const std::vector<int>& arr) {
    if (arr.empty()) {
        return 0;
    }
    int limit = arr[0];
    int sum = 0;
    for (int i = 0; i < limit; ++i) {
        sum += i;
    }
    return sum;
}
