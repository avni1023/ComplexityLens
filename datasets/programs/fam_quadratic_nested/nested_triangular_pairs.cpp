#include <vector>

int count_pairs(int n) {
    int pairs = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            pairs++;
        }
    }
    return pairs;
}
