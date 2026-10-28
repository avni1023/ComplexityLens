int swap_and_sum(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    return a + b;
}
