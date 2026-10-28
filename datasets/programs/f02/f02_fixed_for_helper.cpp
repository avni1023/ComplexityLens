int add_step(int current, int step) {
    return current + step * step;
}

int fixed_for_helper(int initial) {
    int sum = initial;
    for (int idx = 0; idx < 10; ++idx) {
        sum = add_step(sum, idx);
    }
    return sum;
}
