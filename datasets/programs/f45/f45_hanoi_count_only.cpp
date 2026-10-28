// Variant: count-only version — returns the number of moves without printing.
// Structurally distinct: no peg parameters, returns long long.
// T(n) = 2T(n-1) + O(1) -> 2^n - 1 moves = OEXP.
long long hanoi_count_only(int n) {
    if (n <= 0) {
        return 0LL;
    }
    return hanoi_count_only(n - 1) + 1LL + hanoi_count_only(n - 1);
}
