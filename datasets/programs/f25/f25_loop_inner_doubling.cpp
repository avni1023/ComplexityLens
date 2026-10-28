int loop_doubling_inner(int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < n; j *= 2) {
            count += (i ^ j);
        }
    }
    return count;
}
