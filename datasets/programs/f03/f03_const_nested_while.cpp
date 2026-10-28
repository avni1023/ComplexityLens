int while_matrix_const(int base) {
    int sum = base;
    int r = 0;
    while (r < 10) {
        int c = 0;
        while (c < 10) {
            sum += r * 10 + c;
            ++c;
        }
        ++r;
    }
    return sum;
}
