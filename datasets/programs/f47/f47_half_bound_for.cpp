// Seed: for-loop bounded by m = n/2. The rule-hard aspect: the static analyzer
// must recognise that m is a linear alias of n to avoid returning UNKNOWN.
int half_bound_for(int n) {
    int m = n / 2;
    int sum = 0;
    for (int i = 0; i < m; ++i) {
        sum += i;
    }
    return sum;
}
