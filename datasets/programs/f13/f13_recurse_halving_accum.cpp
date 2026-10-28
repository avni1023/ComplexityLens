int halving_accum(int n, int acc) {
    if (n <= 1) {
        return acc;
    }
    return halving_accum(n / 2, acc + (n & 1));
}
