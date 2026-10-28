#include <vector>

// Variant: helper-function split — inner dot product extracted as a helper.
// Structurally distinct: the innermost loop is inside compute_dot().
static long long compute_dot(const std::vector<std::vector<int>>& A,
                              const std::vector<std::vector<int>>& B,
                              int row, int col, int n) {
    long long dot = 0;
    for (int k = 0; k < n; ++k) {
        dot += static_cast<long long>(A[row][k]) * B[k][col];
    }
    return dot;
}

std::vector<std::vector<long long>> matrix_multiply_helper(
    const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B,
    int n) {
    std::vector<std::vector<long long>> C(n, std::vector<long long>(n, 0LL));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = compute_dot(A, B, i, j, n);
        }
    }
    return C;
}
