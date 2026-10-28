#include <vector>

bool is_palindrome_array(const std::vector<int>& vec) {
    if (vec.size() <= 1) return true;
    int low = 0;
    int high = static_cast<int>(vec.size()) - 1;
    while (low < high) {
        if (vec[low] != vec[high]) {
            return false;
        }
        low++;
        high--;
    }
    return true;
}
