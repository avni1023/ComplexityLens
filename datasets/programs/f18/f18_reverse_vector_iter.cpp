#include <vector>
#include <cstddef>

void reverse_vector_in_place(std::vector<int>& vec) {
    if (vec.size() <= 1) return;
    std::size_t l = 0;
    std::size_t r = vec.size() - 1;
    while (l < r) {
        int t = vec[l];
        vec[l] = vec[r];
        vec[r] = t;
        l++;
        r--;
    }
}
