#include <vector>

std::vector<int> prefix_running_sum(const std::vector<int>& a) {
    int n = static_cast<int>(a.size());
    std::vector<int> out(n, 0);
    int running = 0;
    int idx = 0;
    while (idx < n) {
        running += a[idx];
        out[idx] = running;
        idx++;
    }
    return out;
}
