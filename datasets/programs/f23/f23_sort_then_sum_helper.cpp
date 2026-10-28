#include <vector>
#include <algorithm>

void sort_buffer(std::vector<int>& arr) {
    std::sort(arr.begin(), arr.end());
}

int sort_and_prefix_accum(std::vector<int>& data) {
    sort_buffer(data);
    int running = 0;
    for (int x : data) {
        running += x;
    }
    return running;
}
