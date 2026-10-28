#include <vector>
#include <unordered_map>

int frequency_count_explicit_find(const std::vector<int>& nums) {
    std::unordered_map<int, int> freq;
    int n = static_cast<int>(nums.size());
    for (int i = 0; i < n; ++i) {
        auto it = freq.find(nums[i]);
        if (it != freq.end()) {
            it->second += 1;
        } else {
            freq[nums[i]] = 1;
        }
    }
    return freq.empty() ? 0 : freq.begin()->second;
}
