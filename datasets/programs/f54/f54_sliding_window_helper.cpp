static int slide_step(int current, int incoming, int outgoing) {
    return current + incoming - outgoing;
}

int average_sliding_count(const int* arr, int n, int k, int threshold) {
    if (k <= 0 || k > n) return 0;
    int window_sum = 0;
    for (int i = 0; i < k; i++) {
        window_sum += arr[i];
    }
    int count = 0;
    if (window_sum >= threshold) count++;
    int i = k;
    while (i < n) {
        window_sum = slide_step(window_sum, arr[i], arr[i - k]);
        if (window_sum >= threshold) {
            count++;
        }
        i++;
    }
    return count;
}
