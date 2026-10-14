#include <vector>

int find_max_while(const std::vector<int>& arr) {
    if (arr.empty()) return 0;
    int max_val = arr[0];
    int i = 1;
    int n = static_cast<int>(arr.size());
    while (i < n) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
        i++;
    }
    return max_val;
}
