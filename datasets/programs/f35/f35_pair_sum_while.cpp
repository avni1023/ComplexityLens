#include <vector>

int max_pair_sum(const std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    if (n < 2) return 0;
    int best = arr[0] + arr[1];
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n) {
            int s = arr[i] + arr[j];
            if (s > best) {
                best = s;
            }
            ++j;
        }
        ++i;
    }
    return best;
}
