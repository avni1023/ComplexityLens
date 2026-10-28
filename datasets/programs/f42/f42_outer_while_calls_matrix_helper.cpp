// Variant: while-loop calling a helper containing a nested n x n matrix-style loop.
// Structurally distinct: outer uses while, helper uses row/col index naming.
static long long matrix_fill_helper(int rows, int cols) {
    long long checksum = 0;
    int r = 0;
    while (r < rows) {
        int c = 0;
        while (c < cols) {
            checksum += r * cols + c;
            ++c;
        }
        ++r;
    }
    return checksum;
}

long long outer_while_calls_matrix_helper(int n) {
    long long result = 0;
    int k = 0;
    while (k < n) {
        result += matrix_fill_helper(n, n);
        ++k;
    }
    return result;
}
