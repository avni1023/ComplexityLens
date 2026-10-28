int fixed_index_array(const int* a, int size) {
    if (size > 5) {
        return a[0] + a[5];
    }
    return 0;
}
