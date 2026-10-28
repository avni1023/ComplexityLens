#include <vector>
#include <algorithm>

bool contains_value(const std::vector<int>& vec, int target) {
    auto it = std::find(vec.begin(), vec.end(), target);
    return it != vec.end();
}
