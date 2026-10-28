#include <vector>
#include <unordered_map>

int frequency_count_basic(const std::vector<int>& nums) {
    std::unordered_map<int, int> counts;
    for (size_t i = 0; i < nums.size(); ++i) {
        counts[nums[i]]++;
    }
    return static_cast<int>(counts.size());
}
