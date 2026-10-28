static void swap_cells(int* arr, int i, int j) {
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

void reverse_half_loop(int* arr, int n) {
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        swap_cells(arr, i, n - 1 - i);
    }
}
