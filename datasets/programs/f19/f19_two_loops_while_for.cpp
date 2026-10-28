#include <vector>

int sequential_while_for(int n) {
    std::vector<int> buffer(n, 0);
    int i = 0;
    while (i < n) {
        buffer[i] = i * 2;
        ++i;
    }
    int parity_sum = 0;
    for (int j = 0; j < n; ++j) {
        parity_sum += buffer[j] ^ j;
    }
    return parity_sum;
}
