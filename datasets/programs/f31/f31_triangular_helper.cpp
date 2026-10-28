int inner_triangular_scan(int i) {
    int subtotal = 0;
    for (int j = 0; j < i; ++j) {
        subtotal += (j + 1);
    }
    return subtotal;
}

int triangular_with_helper(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += inner_triangular_scan(i);
    }
    return total;
}
