int loop_doubling(int n) {
    int steps = 0;
    for (int i = 1; i < n; i *= 2) {
        steps++;
    }
    return steps;
}
