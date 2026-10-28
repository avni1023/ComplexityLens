int fixed_for_accumulate(int start_val) {
    int total = start_val;
    for (int i = 0; i < 10; ++i) {
        total += i;
    }
    return total;
}
