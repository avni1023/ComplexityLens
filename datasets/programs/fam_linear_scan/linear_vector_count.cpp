#include <vector>
#include <algorithm>

int count_target_occurrences(const std::vector<int>& vec, int target) {
    return static_cast<int>(std::count(vec.begin(), vec.end(), target));
}
