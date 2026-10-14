#include <vector>

int stepped_iteration(int n) {
    const int step = 2;
    int count = 0;
    int i = 0;
    while (i < n) {
        count += i;
        i += step;
    }
    return count;
}
