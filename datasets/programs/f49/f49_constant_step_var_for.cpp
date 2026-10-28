// Variant: for-loop stepping by a const step variable.
// Structurally distinct: for-loop syntax with const step advance.
// ceil(n / 2) iterations -> O(n).
long long constant_step_var_for(int n) {
    const int step = 2;
    long long total = 0;
    for (int i = 0; i < n; i += step) {
        total += (i * 3 + 1);
    }
    return total;
}
