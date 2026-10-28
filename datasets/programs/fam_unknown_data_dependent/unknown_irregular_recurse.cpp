#include <vector>

int irregular_step(int n) {
    if (n <= 1) {
        return 1;
    }
    if (n % 3 == 0) {
        return irregular_step(n - 2) + 1;
    } else {
        return irregular_step(n / 3) + 2;
    }
}
