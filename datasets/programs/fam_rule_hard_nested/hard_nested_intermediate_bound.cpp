#include <vector>

int nested_computed_bounds(int n) {
    int outer_limit = n;
    int total = 0;
    for (int i = 0; i < outer_limit; ++i) {
        int inner_limit = n / 2;
        for (int j = 0; j < inner_limit; ++j) {
            total += (i + j);
        }
    }
    return total;
}
