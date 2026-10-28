// Variant: helper-function split — step size is provided by a helper function.
// Structurally distinct: step size obtained via helper call.
// ceil(n / 2) iterations -> O(n).
static int get_stride() {
    return 2;
}

long long constant_step_var_helper(int n) {
    int stride = get_stride();
    int idx = 0;
    long long acc = 0;
    while (idx < n) {
        acc += idx;
        idx += stride;
    }
    return acc;
}
