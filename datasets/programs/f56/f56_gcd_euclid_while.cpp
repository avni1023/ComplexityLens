// Seed: classic Euclid's GCD with while-loop.
// The remainder sequence shrinks geometrically: O(log n) steps.
int gcd_euclid_while(int a, int b) {
    while (b != 0) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}
