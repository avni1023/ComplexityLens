#include <vector>

int sum_row(const std::vector<int>& row, int n) {
    int row_sum = 0;
    for (int j = 0; j < n; ++j) {
        row_sum += row[j];
    }
    return row_sum;
}

int sum_matrix_with_helper(const std::vector<std::vector<int>>& grid, int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += sum_row(grid[i], n);
    }
    return total;
}
