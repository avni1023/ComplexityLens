// Fixture 08: STL std::sort usage
// Expected: CallExpr to std::sort, DeclRefExpr to algorithm functions
#include <algorithm>
#include <vector>

void sort_vec(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
}
