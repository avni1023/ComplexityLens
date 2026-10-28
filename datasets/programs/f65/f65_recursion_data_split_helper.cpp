#include <vector>

int choose_split_from_data(const std::vector<int>& vec, int left, int right) {
    int offset = vec[left];
    int candidate = left + (offset > 0 ? offset : 1);
    if (candidate >= right) {
        candidate = right - 1;
    }
    return candidate;
}

int recursive_split_helper(const std::vector<int>& vec, int l, int r) {
    if (l >= r) {
        return 0;
    }
    int split = choose_split_from_data(vec, l, r);
    return 1 + recursive_split_helper(vec, l, split) + recursive_split_helper(vec, split + 1, r);
}
