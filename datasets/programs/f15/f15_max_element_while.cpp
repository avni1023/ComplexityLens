#include <vector>

int max_while_scan(const std::vector<int>& a) {
    int n = static_cast<int>(a.size());
    if (n == 0) return 0;
    int max_elem = a[0];
    int idx = 1;
    while (idx < n) {
        if (a[idx] > max_elem) {
            max_elem = a[idx];
        }
        idx++;
    }
    return max_elem;
}
