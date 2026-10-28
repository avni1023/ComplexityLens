int triangular_nested_for(int n) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            count += (i + j);
        }
    }
    return count;
}
