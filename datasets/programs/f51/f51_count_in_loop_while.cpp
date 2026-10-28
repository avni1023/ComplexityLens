#include <vector>
#include <algorithm>

// Variant: while-loop calling std::count for values 0 to n-1.
// Structurally distinct: while-loop syntax with integer index stepping.
// n * n = O(n^2).
long long count_in_loop_while(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    long long matched = 0;
    int val = 0;
    while (val < n) {
        matched += std::count(v.begin(), v.end(), val);
        ++val;
    }
    return matched;
}
