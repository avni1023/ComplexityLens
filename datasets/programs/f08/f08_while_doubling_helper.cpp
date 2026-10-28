static int count_doubling_steps(int limit) {
    int cur = 1;
    int acc = 0;
    while (cur < limit) {
        acc += cur;
        cur = cur * 2;
    }
    return acc;
}

int compute_doubling(int n) {
    if (n <= 1) {
        return 0;
    }
    return count_doubling_steps(n);
}
