int grid_accumulate(int init) {
    int total = init;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            total += (i + j);
        }
    }
    return total;
}
