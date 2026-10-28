#include <vector>

// Variant: merge logic extracted into a standalone helper function that takes
// the two halves as separate vectors (different helper-function split structure).
static std::vector<int> merge_two_halves(const std::vector<int>& left,
                                          const std::vector<int>& right) {
    std::vector<int> result;
    result.reserve(left.size() + right.size());
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(left.size()) && j < static_cast<int>(right.size())) {
        if (left[i] <= right[j]) {
            result.push_back(left[i++]);
        } else {
            result.push_back(right[j++]);
        }
    }
    while (i < static_cast<int>(left.size())) {
        result.push_back(left[i++]);
    }
    while (j < static_cast<int>(right.size())) {
        result.push_back(right[j++]);
    }
    return result;
}

std::vector<int> mergesort_helper_merge(std::vector<int> arr) {
    int n = static_cast<int>(arr.size());
    if (n <= 1) {
        return arr;
    }
    int mid = n / 2;
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());
    left = mergesort_helper_merge(left);
    right = mergesort_helper_merge(right);
    return merge_two_halves(left, right);
}
