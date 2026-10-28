int fixed_while_accumulate(int base) {
    int total = base;
    int i = 0;
    while (i < 10) {
        total += (i * 2 + 1);
        ++i;
    }
    return total;
}
