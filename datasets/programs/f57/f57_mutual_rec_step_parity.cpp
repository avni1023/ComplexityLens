int parity_odd_step(int n, int acc);

int parity_even_step(int n, int acc) {
    if (n <= 0) {
        return acc;
    }
    return parity_odd_step(n - 1, acc + 1);
}

int parity_odd_step(int n, int acc) {
    if (n <= 0) {
        return acc;
    }
    return parity_even_step(n - 1, acc + 3);
}
