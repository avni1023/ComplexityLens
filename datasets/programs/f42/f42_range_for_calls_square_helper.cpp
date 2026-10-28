#include <vector>

// Variant: range-for over a size-n vector calling a helper with nested for-loops.
// Structurally distinct: outer uses range-for with container, helper uses separate n parameter.
static long long square_sum_helper(int sz) {
    long long s = 0;
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            s += (i + 1) * (j + 1);
        }
    }
    return s;
}

long long range_for_calls_square_helper(const std::vector<int>& items) {
    int n = static_cast<int>(items.size());
    long long total = 0;
    for (int elem : items) {
        total += square_sum_helper(n) + elem;
    }
    return total;
}
