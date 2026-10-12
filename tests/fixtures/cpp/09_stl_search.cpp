// Fixture 09: STL std::find (linear search)
// Expected: CallExpr to std::find, iterator return type visible in AST
#include <algorithm>
#include <vector>

bool contains(const std::vector<int>& v, int val) {
    return std::find(v.begin(), v.end(), val) != v.end();
}
