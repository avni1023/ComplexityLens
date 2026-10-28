long long power_opt(long long x, int n) {
    if (n == 0) return 1;
    if (n == 1) return x;
    if (x == 0 || x == 1) return x;
    long long result = 1;
    long long cur = x;
    int exp = n;
    while (exp > 0) {
        if (exp % 2 != 0) {
            result *= cur;
        }
        cur *= cur;
        exp /= 2;
    }
    return result;
}
