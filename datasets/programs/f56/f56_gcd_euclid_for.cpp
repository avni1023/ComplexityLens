// Variant: Euclid's algorithm expressed as a for-loop with no fixed bound.
// The loop iterates until b == 0; the for-update performs the modulo reduction.
// Structurally distinct: uses for-loop idiom with update in the body.
int gcd_euclid_for(int a, int b) {
    int steps = 0;
    for (; b != 0; ++steps) {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}
