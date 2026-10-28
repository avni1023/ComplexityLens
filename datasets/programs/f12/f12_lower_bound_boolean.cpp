#include <vector>
#include <algorithm>

bool contains_value_lower_bound(const std::vector<int>& arr, int target) {
    auto it = std::lower_bound(arr.begin(), arr.end(), target);
    return (it != arr.end() && *it == target);
}
