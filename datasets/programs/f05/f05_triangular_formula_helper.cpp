long long multiply_terms(long long a, long long b) {
    return a * b;
}

long long triangular_formula_helper(int n) {
    if (n <= 0) {
        return 0;
    }
    long long term1 = n;
    long long term2 = static_cast<long long>(n) + 1;
    return multiply_terms(term1, term2) / 2;
}
