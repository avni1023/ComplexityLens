int range_halve_worker(int low, int high) {
    if (high - low <= 1) {
        return low;
    }
    int mid = low + (high - low) / 2;
    return range_halve_worker(low, mid) + 1;
}

int recursive_log_range(int n) {
    if (n <= 1) {
        return 0;
    }
    return range_halve_worker(1, n);
}
