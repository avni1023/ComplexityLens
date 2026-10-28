#include <cmath>

// Variant: helper-function split — iterative approximation step is in a helper.
// Structurally distinct: step calculation extracted to helper function.
// UNKNOWN / termination.
static double next_estimate(double x, double S) {
    return 0.5 * (x + S / x);
}

double float_convergence_helper(double S, double eps) {
    if (S <= 0.0 || eps <= 0.0) {
        return 0.0;
    }
    double curr = S;
    double prev = 0.0;
    while (std::fabs(curr - prev) > eps) {
        prev = curr;
        curr = next_estimate(curr, S);
    }
    return curr;
}
