#include <vector>

extern int external_worker(int val);

int run_external_loop(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += external_worker(i);
    }
    return total;
}
