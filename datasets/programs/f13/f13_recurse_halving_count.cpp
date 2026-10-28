int count_halvings(int n) {
    if (n <= 1) {
        return 0;
    }
    return 1 + count_halvings(n / 2);
}
