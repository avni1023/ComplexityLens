#include <vector>

// Variant: triple nested while-loops counting all triples equal to target.
// Structurally distinct: uses while loops with manual index management.
// T(n) = Theta(n^3) -> ON3.
int triple_sum_while(const std::vector<int>& v, int target) {
    int n = static_cast<int>(v.size());
    int count = 0;
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n) {
            int k = j + 1;
            while (k < n) {
                if (v[i] + v[j] + v[k] == target) {
                    ++count;
                }
                ++k;
            }
            ++j;
        }
        ++i;
    }
    return count;
}
