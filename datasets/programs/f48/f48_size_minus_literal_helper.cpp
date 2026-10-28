#include <vector>

// Variant: helper-function split — a helper computes the adjusted limit from size.
// Structurally distinct: loop body and limit derivation are in separate functions.
static int adjusted_limit(const std::vector<int>& v, int offset) {
    int n = static_cast<int>(v.size());
    return n > offset ? n - offset : 0;
}

int size_minus_literal_helper(const std::vector<int>& v) {
    int lim = adjusted_limit(v, 2);
    int product = 1;
    for (int i = 0; i < lim; ++i) {
        if (v[i] != 0) {
            product *= v[i];
        }
    }
    return product;
}
