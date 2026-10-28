#include <vector>

int dac_sum(const std::vector<int>& v, int lo, int hi) {
    if (lo > hi) {
        return 0;
    }
    if (lo == hi) {
        return v[lo];
    }
    int mid = lo + (hi - lo) / 2;
    int left_sum = dac_sum(v, lo, mid);
    int right_sum = dac_sum(v, mid + 1, hi);
    return left_sum + right_sum;
}
