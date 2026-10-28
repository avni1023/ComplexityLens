long long triangular_formula_direct(int n) {
    if (n <= 0) {
        return 0;
    }
    long long val = n;
    return val * (val + 1) / 2;
}
