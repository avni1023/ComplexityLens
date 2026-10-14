#include <vector>

int fixed_iterations(int val) {
    int acc = val;
    for (int i = 0; i < 10; ++i) {
        acc += i * 2;
    }
    return acc;
}
