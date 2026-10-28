long long triangular_formula_bitwise(int n) {
    if (n <= 0) {
        return 0;
    }
    long long val = n;
    return (val * (val + 1)) >> 1;
}
