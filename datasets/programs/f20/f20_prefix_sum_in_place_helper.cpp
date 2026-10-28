static int add_step(int prev, int curr) {
    return prev + curr;
}

void compute_in_place_prefix(int* arr, int n) {
    if (n <= 1) return;
    for (int i = 1; i < n; i++) {
        arr[i] = add_step(arr[i - 1], arr[i]);
    }
}
