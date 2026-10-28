int traverse_grid_while(int n) {
    int total = 0;
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n) {
            total += (i + j);
            ++j;
        }
        ++i;
    }
    return total;
}
