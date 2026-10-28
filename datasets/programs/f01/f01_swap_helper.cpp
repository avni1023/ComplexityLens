static void swap_references(int& x, int& y) {
    int t = x;
    x = y;
    y = t;
}

int swap_and_sum_helper(int a, int b) {
    swap_references(a, b);
    return a + b;
}
