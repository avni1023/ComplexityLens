// Fixture 07: Direct recursion (fibonacci)
// Expected: FunctionDecl with a CallExpr whose callee == same FunctionDecl
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
