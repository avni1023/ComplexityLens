bool has_duplicate_pairs_while(const int* arr, int n) {
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n) {
            if (arr[i] == arr[j]) {
                return true;
            }
            ++j;
        }
        ++i;
    }
    return false;
}
