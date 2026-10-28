#include <vector>
#include <unordered_map>

int find_most_frequent_count(const std::vector<int>& items) {
    std::unordered_map<int, int> freq_map;
    for (int val : items) {
        freq_map[val] += 1;
    }
    int max_freq = 0;
    for (const auto& pair : freq_map) {
        if (pair.second > max_freq) {
            max_freq = pair.second;
        }
    }
    return max_freq;
}
