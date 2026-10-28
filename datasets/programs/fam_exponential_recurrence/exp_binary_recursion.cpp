#include <vector>

int count_paths(int n) {
    if (n <= 0) {
        return 1;
    }
    return count_paths(n - 1) + count_paths(n - 1);
}
