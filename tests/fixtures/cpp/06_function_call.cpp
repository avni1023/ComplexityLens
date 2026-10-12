// Fixture 06: Function calling another function
// Expected: 2x FunctionDecl, CallExpr with callee reference
int helper(int x) {
    return x * 2;
}

int caller(int n) {
    return helper(n) + helper(n - 1);
}
