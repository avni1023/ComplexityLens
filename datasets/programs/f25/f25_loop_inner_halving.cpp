int loop_halving_direct(int n) {
    int ops = 0;
    for (int i = 0; i < n; ++i) {
        int j = n;
        while (j > 0) {
            ops += (i + j) & 1;
            j /= 2;
        }
    }
    return ops;
}
