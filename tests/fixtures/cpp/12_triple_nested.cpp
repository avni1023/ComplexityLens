// Fixture 12: Triple-nested loops (O(n^3) pattern)
// Expected: 3x ForStmt at depths 1, 2, 3 relative to function body
int cube_sum(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                total += i + j + k;
            }
        }
    }
    return total;
}
