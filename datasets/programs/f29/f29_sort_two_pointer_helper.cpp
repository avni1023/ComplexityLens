#include <vector>
#include <algorithm>

int count_pairs_less_than(const std::vector<int>& sorted_vec, int threshold) {
    int count = 0;
    int i = 0;
    int j = static_cast<int>(sorted_vec.size()) - 1;
    while (i < j) {
        if (sorted_vec[i] + sorted_vec[j] < threshold) {
            count += (j - i);
            ++i;
        } else {
            --j;
        }
    }
    return count;
}

int sort_and_count_pairs(std::vector<int>& data, int thresh) {
    std::sort(data.begin(), data.end());
    return count_pairs_less_than(data, thresh);
}
