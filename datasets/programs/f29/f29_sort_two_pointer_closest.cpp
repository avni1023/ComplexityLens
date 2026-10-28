#include <vector>
#include <algorithm>

int find_closest_pair_sum(std::vector<int>& v, int target) {
    if (v.size() < 2) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    int l = 0;
    int r = static_cast<int>(v.size()) - 1;
    int best_diff = 2000000000;
    int closest_sum = 0;
    while (l < r) {
        int current_sum = v[l] + v[r];
        int diff = current_sum - target;
        if (diff < 0) diff = -diff;
        if (diff < best_diff) {
            best_diff = diff;
            closest_sum = current_sum;
        }
        if (current_sum < target) {
            ++l;
        } else {
            --r;
        }
    }
    return closest_sum;
}
