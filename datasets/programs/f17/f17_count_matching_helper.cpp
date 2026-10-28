#include <vector>
#include <cstddef>

bool is_positive_and_even(int x) {
    return (x > 0) && (x % 2 == 0);
}

int count_matching_helper(const std::vector<int>& v) {
    int count = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (is_positive_and_even(v[i])) {
            count++;
        }
    }
    return count;
}
