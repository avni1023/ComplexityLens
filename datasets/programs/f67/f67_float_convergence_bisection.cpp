#include <cmath>

// Variant: interval bisection search for root of f(x) = x^2 - target.
// Structurally distinct: bisection interval halving on continuous real line.
// UNKNOWN / termination: halts when interval width |high - low| <= eps.
double float_convergence_bisection(double target, double eps) {
    if (target < 0.0 || eps <= 0.0) {
        return 0.0;
    }
    double low = 0.0;
    double high = (target > 1.0) ? target : 1.0;
    while (std::fabs(high - low) > eps) {
        double mid = 0.5 * (low + high);
        if (mid * mid < target) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return 0.5 * (low + high);
}
