#include <vector>

int front_back_iter(const std::vector<int>& v) {
    if (v.size() < 1) {
        return 0;
    }
    auto it_first = v.begin();
    auto it_last = v.end() - 1;
    return *it_first + *it_last;
}
