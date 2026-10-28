#include <vector>

// Seed: standard triple-nested for-loop matrix multiplication.
std::vector<std::vector<long long>> matrix_multiply_basic(
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    int n) {
    std::vector<std::vector<long long>> C(n, std::vector<long long>(n, 0LL));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += static_cast<long long>(A[i][k]) * B[k][j];
            }
        }
    }
    return C;
}
