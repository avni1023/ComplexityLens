#include <vector>
#include <algorithm>

int query_single_bound(const std::vector<int>& sorted_arr, int target) {
    auto it = std::lower_bound(sorted_arr.begin(), sorted_arr.end(), target);
    return static_cast<int>(it - sorted_arr.begin());
}

int sort_and_helper_queries(std::vector<int>& data) {
    std::sort(data.begin(), data.end());
    int total_rank = 0;
    for (int val : data) {
        total_rank += query_single_bound(data, val);
    }
    return total_rank;
}
