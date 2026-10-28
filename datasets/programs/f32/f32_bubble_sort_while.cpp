#include <vector>

void bubble_sort_while(std::vector<int>& arr) {
    int n = static_cast<int>(arr.size());
    int i = 0;
    while (i < n) {
        int j = 0;
        while (j < n - 1) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
            ++j;
        }
        ++i;
    }
}
