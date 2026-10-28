int triangular_nested_while(int n) {
    int count = 0;
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < i) {
            count += (i ^ j);
            ++j;
        }
        ++i;
    }
    return count;
}
