// Variant: ternary expression base case without intermediate variables.
// Structurally distinct: single compound return expression.
// T(n) = 3T(n-1) + O(1) -> OEXP.
long long three_calls_ternary(int n) {
    return (n <= 0)
        ? 1LL
        : (three_calls_ternary(n - 1) +
           three_calls_ternary(n - 1) +
           three_calls_ternary(n - 1));
}
