// Fixture 04: Nested for loops (O(n^2) pattern)
// Expected: 2x ForStmt (outer/inner nesting detectable by depth)
int matrix_sum(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            total += i * j;
        }
    }
    return total;
}
