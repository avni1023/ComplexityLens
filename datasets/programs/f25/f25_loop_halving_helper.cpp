int count_log_steps(int limit) {
    int steps = 0;
    int val = limit;
    while (val > 0) {
        steps++;
        val /= 2;
    }
    return steps;
}

int loop_with_log_helper(int n) {
    int total = 0;
    int i = 0;
    while (i < n) {
        total += count_log_steps(n);
        ++i;
    }
    return total;
}
