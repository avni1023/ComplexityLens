int doubling_shift(int n) {
    if (n <= 0) {
        return 0;
    }
    int count = 0;
    for (long long p = 1; p < n; p <<= 1) {
        count++;
    }
    return count;
}
