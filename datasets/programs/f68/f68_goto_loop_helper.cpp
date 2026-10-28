int step_acc(int current, int val) {
    return current + val;
}

int goto_loop_helper(int n) {
    int total = 0;
    int idx = 0;
repeat_section:
    if (idx < n) {
        total = step_acc(total, idx);
        idx++;
        goto repeat_section;
    }
    return total;
}
