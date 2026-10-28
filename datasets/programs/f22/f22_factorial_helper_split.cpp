long long factorial_worker(int current, int bound) {
    if (current > bound) {
        return 1;
    }
    return current * factorial_worker(current + 1, bound);
}

long long factorial_upward(int n) {
    if (n <= 1) {
        return 1;
    }
    return factorial_worker(1, n);
}
