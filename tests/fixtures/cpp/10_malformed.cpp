// Fixture 10: Malformed C++ — missing closing brace and semicolon
// Expected: Parser reports diagnostics with line/col, partial AST still produced
int broken(int x {
    return x +
