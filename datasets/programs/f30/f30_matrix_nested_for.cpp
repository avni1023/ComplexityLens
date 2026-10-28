#include <vector>

int sum_matrix_cells(const std::vector<std::vector<int>>& mat, int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            total += mat[i][j];
        }
    }
    return total;
}
