// Variant: while-loop with the bound stored as m = n / 2.
// Structurally distinct from seed: uses while with explicit counter.
int half_bound_while(int n) {
    int m = n / 2;
    int acc = 0;
    int i = 0;
    while (i < m) {
        acc += i * 2 + 1;
        ++i;
    }
    return acc;
}
