long long sum_array_while(const int* arr, int n) {
    long long total = 0;
    int i = 0;
    while (i < n) {
        total += arr[i];
        ++i;
    }
    return total;
}
