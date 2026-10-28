#include <vector>

// Variant: helper-function split — the half-bound computation is done by a helper
// that returns m = n/2, and the main function loops up to that returned value.
static int compute_half(int n) {
    return n / 2;
}

int half_bound_helper(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    int limit = compute_half(n);
    int result = 0;
    for (int i = 0; i < limit; ++i) {
        result += v[i];
    }
    return result;
}
