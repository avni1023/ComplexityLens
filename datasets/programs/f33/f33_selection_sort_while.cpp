#include <vector>

void selection_sort_while(std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    int i = 0;
    while (i < n) {
        int min_pos = i;
        int j = i + 1;
        while (j < n) {
            if (v[j] < v[min_pos]) {
                min_pos = j;
            }
            ++j;
        }
        int t = v[i];
        v[i] = v[min_pos];
        v[min_pos] = t;
        ++i;
    }
}
