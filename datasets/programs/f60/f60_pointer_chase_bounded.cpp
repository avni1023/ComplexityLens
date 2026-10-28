int pointer_chase_array(const int* jumps, int n, int start_idx, int end_idx) {
    int hops = 0;
    int pos = start_idx;
    while (pos != end_idx && pos >= 0 && pos < n) {
        pos = jumps[pos];
        hops++;
    }
    return hops;
}
