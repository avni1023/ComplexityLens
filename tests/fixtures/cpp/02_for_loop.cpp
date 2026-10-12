// Fixture 02: Simple for loop
// Expected: FunctionDecl, ForStmt, VarDecl, BinaryOperator, UnaryOperator
int sum_for(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += i;
    }
    return total;
}
