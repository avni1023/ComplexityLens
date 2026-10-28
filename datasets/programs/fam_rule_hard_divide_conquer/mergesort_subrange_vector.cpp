#include <vector>

void merge_two_halves(std::vector<int>& v, int left, int mid, int right) {
    std::vector<int> out;
    int i = left;
    int j = mid;
    while (i < mid && j < right) {
        if (v[i] <= v[j]) {
            out.push_back(v[i++]);
        } else {
            out.push_back(v[j++]);
        }
    }
    while (i < mid) out.push_back(v[i++]);
    while (j < right) out.push_back(v[j++]);
    for (size_t k = 0; k < out.size(); ++k) {
        v[left + k] = out[k];
    }
}

void divide_and_conquer_sort(std::vector<int>& v, int left, int right) {
    if (right - left <= 1) return;
    int mid = left + (right - left) / 2;
    divide_and_conquer_sort(v, left, mid);
    divide_and_conquer_sort(v, mid, right);
    merge_two_halves(v, left, mid, right);
}
