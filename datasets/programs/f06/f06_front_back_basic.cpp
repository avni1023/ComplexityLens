#include <vector>

int front_plus_back(const std::vector<int>& v) {
    if (v.empty()) {
        return 0;
    }
    return v.front() + v.back();
}
