bool linear_search_while(const int* arr, int n, int key) {
    int idx = 0;
    while (idx < n) {
        if (arr[idx] == key) {
            return true;
        }
        idx++;
    }
    return false;
}
