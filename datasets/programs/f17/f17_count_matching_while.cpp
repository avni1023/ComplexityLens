int count_positive_while(const int* arr, int n) {
    int count = 0;
    int i = 0;
    while (i < n) {
        if (arr[i] > 0) {
            count++;
        }
        i++;
    }
    return count;
}
