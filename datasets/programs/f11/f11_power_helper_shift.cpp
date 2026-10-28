static long long mod_mult(long long a, long long b, long long mod) {
    return (a * b) % mod;
}

long long power_modulo(long long base, int n) {
    const long long MOD = 1000000007;
    long long ans = 1;
    base %= MOD;
    for (; n > 0; n >>= 1) {
        if (n & 1) {
            ans = mod_mult(ans, base, MOD);
        }
        base = mod_mult(base, base, MOD);
    }
    return ans;
}
