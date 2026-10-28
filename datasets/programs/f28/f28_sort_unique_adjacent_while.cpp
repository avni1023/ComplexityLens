#include <vector>
#include <algorithm>

int count_unique_while_scan(std::vector<int>& arr) {
    if (arr.empty()) {
        return 0;
    }
    std::sort(arr.begin(), arr.end());
    int distinct = 1;
    int idx = 1;
    int n = static_cast<int>(arr.size());
    while (idx < n) {
        if (arr[idx] != arr[idx - 1]) {
            ++distinct;
        }
        ++idx;
    }
    return distinct;
}
