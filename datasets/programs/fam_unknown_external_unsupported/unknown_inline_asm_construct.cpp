#include <vector>

int perform_asm_op(int n) {
    int result = 0;
    for (int i = 0; i < n; ++i) {
        __asm__ volatile ("nop");
        result += i;
    }
    return result;
}
