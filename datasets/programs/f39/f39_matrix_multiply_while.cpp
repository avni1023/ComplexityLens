#include <vector>

// Variant: while-loop style for all three levels — structurally distinct loop syntax.
std::vector<std::vector<long long>> matrix_multiply_while(
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    int n) {
    std::vector<std::vector<long long>> C(n, std::vector<long long>(n, 0LL));
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n) {
            long long dot = 0;
            int k = 0;
            while (k < n) {
                dot += static_cast<long long>(A[i][k]) * B[k][j];
                ++k;
            }
            C[i][j] = dot;
            ++j;
        }
        ++i;
    }
    return C;
}
