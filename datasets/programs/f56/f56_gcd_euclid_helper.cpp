// Variant: helper-function split — the single modulo step is extracted into a helper.
// Main loop invokes the helper repeatedly until the remainder reaches zero.
// Structurally distinct: helper-function decomposition.
static int euclid_step(int a, int b) {
    return a % b;
}

int gcd_euclid_helper(int a, int b) {
    while (b != 0) {
        int r = euclid_step(a, b);
        a = b;
        b = r;
    }
    return a;
}
