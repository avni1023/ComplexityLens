int swap_and_sum_arithmetic(int a, int b) {
    if (a == b) {
        return a + b;
    }
    a = a + b;
    b = a - b;
    a = a - b;
    return a + b;
}
