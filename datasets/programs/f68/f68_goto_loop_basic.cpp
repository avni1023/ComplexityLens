int goto_loop_basic(int n) {
    int sum = 0;
    int i = 0;
loop_head:
    if (i < n) {
        sum += i;
        ++i;
        goto loop_head;
    }
    return sum;
}
