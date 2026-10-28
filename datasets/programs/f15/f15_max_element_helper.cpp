static int larger_of(int a, int b) {
    return (a > b) ? a : b;
}

int array_maximum(const int* arr, int n) {
    if (n <= 0) return 0;
    int current_max = arr[0];
    for (int i = 1; i < n; i++) {
        current_max = larger_of(current_max, arr[i]);
    }
    return current_max;
}
