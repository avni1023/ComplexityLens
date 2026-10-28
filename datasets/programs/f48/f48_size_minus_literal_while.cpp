#include <vector>

// Variant: while-loop with the n-2 bound stored as a computed integer variable.
// Structurally distinct: explicit int limit = n - 2 used in while condition.
int size_minus_literal_while(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    if (n < 2) {
        return 0;
    }
    int limit = n - 2;
    int count = 0;
    int i = 0;
    while (i < limit) {
        if (v[i] > v[i + 1]) {
            ++count;
        }
        ++i;
    }
    return count;
}
