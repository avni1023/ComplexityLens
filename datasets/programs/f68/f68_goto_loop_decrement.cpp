int goto_loop_decrement(int n) {
    int count = 0;
start_label:
    if (n > 0) {
        count += n;
        n -= 2;
        goto start_label;
    }
    return count;
}
