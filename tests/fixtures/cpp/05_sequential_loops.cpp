// Fixture 05: Sequential (non-nested) loops — O(n) not O(n^2)
// Expected: 2x ForStmt at same depth in the same function body
int two_passes(int n) {
    int a = 0;
    for (int i = 0; i < n; ++i) {
        a += i;
    }
    int b = 0;
    for (int j = 0; j < n; ++j) {
        b -= j;
    }
    return a + b;
}
