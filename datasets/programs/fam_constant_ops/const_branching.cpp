#include <vector>

int branch_check(int x, int y) {
    if (x > y) {
        return x - y;
    } else {
        return y - x;
    }
}
