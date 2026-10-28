#include <vector>
#include <algorithm>

int scan_adjacent_changes(const std::vector<int>& sorted_items) {
    if (sorted_items.empty()) {
        return 0;
    }
    int changes = 1;
    for (size_t i = 1; i < sorted_items.size(); ++i) {
        if (sorted_items[i] != sorted_items[i - 1]) {
            changes++;
        }
    }
    return changes;
}

int sort_and_scan_driver(std::vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    return scan_adjacent_changes(nums);
}
