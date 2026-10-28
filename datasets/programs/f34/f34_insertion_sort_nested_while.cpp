#include <vector>

void insertion_sort_pure_while(std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    int i = 1;
    while (i < n) {
        int val = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > val) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = val;
        ++i;
    }
}
