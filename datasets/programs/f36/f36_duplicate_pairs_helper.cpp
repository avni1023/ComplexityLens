#include <vector>

bool occurs_after(const std::vector<int>& v, int index, int n) {
    for (int j = index + 1; j < n; ++j) {
        if (v[j] == v[index]) {
            return true;
        }
    }
    return false;
}

bool has_duplicate_pairs_helper(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    for (int i = 0; i < n; ++i) {
        if (occurs_after(v, i, n)) {
            return true;
        }
    }
    return false;
}
