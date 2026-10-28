long long factorial_tail_rec(int n, long long acc) {
    if (n <= 1) {
        return acc;
    }
    return factorial_tail_rec(n - 1, acc * n);
}

long long factorial_tail_driver(int n) {
    return factorial_tail_rec(n, 1);
}
