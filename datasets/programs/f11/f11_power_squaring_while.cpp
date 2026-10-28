long long power_squaring(long long base, int n) {
    long long res = 1;
    while (n > 0) {
        if (n % 2 == 1) {
            res = res * base;
        }
        base = base * base;
        n /= 2;
    }
    return res;
}
