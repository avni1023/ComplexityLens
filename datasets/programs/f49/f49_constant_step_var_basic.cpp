// Seed: while-loop stepping by a constant stored in a local variable step = 2.
// ceil(n / 2) iterations of O(1) body -> O(n).
long long constant_step_var_basic(int n) {
    int step = 2;
    int i = 0;
    long long sum = 0;
    while (i < n) {
        sum += i;
        i += step;
    }
    return sum;
}
