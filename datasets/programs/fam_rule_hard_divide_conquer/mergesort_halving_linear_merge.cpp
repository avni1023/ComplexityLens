#include <vector>

void merge_pass(std::vector<int>& a, int lo, int hi) {
    int mid = (lo + hi) / 2;
    std::vector<int> res;
    int i = lo, j = mid + 1;
    while (i <= mid && j <= hi) {
        if (a[i] < a[j]) res.push_back(a[i++]);
        else res.push_back(a[j++]);
    }
    while (i <= mid) res.push_back(a[i++]);
    while (j <= hi) res.push_back(a[j++]);
    for (int k = 0; k < static_cast<int>(res.size()); ++k) {
        a[lo + k] = res[k];
    }
}

void recursive_halve(std::vector<int>& a, int lo, int hi) {
    if (lo >= hi) return;
    int mid = (lo + hi) / 2;
    recursive_halve(a, lo, mid);
    recursive_halve(a, mid + 1, hi);
    merge_pass(a, lo, hi);
}
