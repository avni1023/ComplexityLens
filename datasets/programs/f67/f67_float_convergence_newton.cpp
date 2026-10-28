#include <cmath>

// Seed: Newton-Raphson approximation for square root.
// Loop terminates when successive iterates differ by less than eps.
// UNKNOWN / termination: continuous floating-point convergence cannot be bounded statically.
double float_convergence_newton(double S, double eps) {
    if (S <= 0.0 || eps <= 0.0) {
        return 0.0;
    }
    double x = S;
    double prev = 0.0;
    while (std::fabs(x - prev) > eps) {
        prev = x;
        x = 0.5 * (x + S / x);
    }
    return x;
}
